import cv2
from face_detector import FaceDetector
from deepface_classifier import DeepFaceCognitiveClassifier
from gaze_detector import GazeDetector


def main():
    print("[INFO] Inicializando sistema con DeepFace + Gaze Tracking...")
    print("[INFO] NOTA: Primera ejecución descargará modelos")
    
    face_detector = FaceDetector()
    emotion_classifier = DeepFaceCognitiveClassifier()
    gaze_detector = GazeDetector()
    
    if not emotion_classifier.model_loaded:
        print("[ERROR] No se pudo cargar el modelo de emociones. Ejecuta:")
        print("  pip install -r requirements_deepface.txt")
        return
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la cámara")
        return
    
    print("[INFO] Sistema iniciado correctamente")
    print("[INFO] Presiona 'q' para salir")
    print("[INFO] Presiona 'd' para ver detalles de emociones")
    print("[INFO] Presiona 'g' para ver análisis de mirada")
    
    state_colors = {
        'concentrado': (0, 255, 0),
        'distraido': (0, 165, 255),
        'frustrado': (0, 0, 255),
        'entendiendo': (255, 255, 0),
        'desconocido': (128, 128, 128)
    }
    
    frame_count = 0
    show_details = False
    show_gaze = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer el frame")
            break
        
        bbox = face_detector.detect(frame)
        
        if bbox and frame_count % 5 == 0:
            face_crop = face_detector.crop_face(frame, bbox)
            
            state, confidence, emotion, all_emotions = emotion_classifier.predict(face_crop)
            
            looking_at_camera, gaze_direction, eye_count = gaze_detector.analyze_gaze(face_crop)
            
            if show_gaze:
                frame_display = gaze_detector.draw_eye_analysis(
                    frame, bbox, looking_at_camera, gaze_direction, eye_count
                )
            else:
                frame_display = face_detector.draw_bbox(frame, bbox)
            
            color = state_colors.get(state, (255, 255, 255))
            
            cv2.putText(frame_display, f"Estado: {state.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.putText(frame_display, f"Confianza: {confidence:.0%}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.putText(frame_display, f"Emocion: {emotion}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            gaze_status = "SI" if looking_at_camera else "NO"
            gaze_color = (0, 255, 0) if looking_at_camera else (0, 0, 255)
            cv2.putText(frame_display, f"Mirando camara: {gaze_status}", (10, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 2)
            
            if show_details and all_emotions:
                y_pos = 170
                for emo, score in all_emotions.items():
                    text = f"{emo}: {score:.1f}%"
                    cv2.putText(frame_display, text, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    y_pos += 20
            
            if frame_count % 30 == 0:
                print(f"[PREDICCION] Estado: {state.upper()} | Confianza: {confidence:.0%} | Emoción: {emotion} | Mirando: {gaze_status} | Dirección: {gaze_direction}")
                if show_details and all_emotions:
                    print(f"  Detalles: {all_emotions}")
        elif bbox:
            frame_display = face_detector.draw_bbox(frame, bbox)
        else:
            frame_display = frame
            cv2.putText(frame_display, "Sin rostro detectado", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Analisis Cognitivo + Gaze Tracking', frame_display)
        
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_details = not show_details
            print(f"[INFO] Detalles de emociones: {'activados' if show_details else 'desactivados'}")
        elif key == ord('g'):
            show_gaze = not show_gaze
            print(f"[INFO] Análisis de mirada: {'activado' if show_gaze else 'desactivado'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Sistema detenido")


if __name__ == "__main__":
    main()
