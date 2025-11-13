import cv2
from face_detector import FaceDetector
from deepface_classifier import DeepFaceCognitiveClassifier


def main():
    print("[INFO] Inicializando sistema con DeepFace...")
    print("[INFO] NOTA: Primera ejecución descargará modelos (~100MB)")
    
    detector = FaceDetector()
    classifier = DeepFaceCognitiveClassifier()
    
    if not classifier.model_loaded:
        print("[ERROR] No se pudo cargar el modelo. Ejecuta:")
        print("  pip install -r requirements_deepface.txt")
        return
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la cámara")
        return
    
    print("[INFO] Sistema iniciado correctamente")
    print("[INFO] Presiona 'q' para salir")
    print("[INFO] Presiona 'd' para ver detalles de emociones")
    
    state_colors = {
        'concentrado': (0, 255, 0),
        'distraido': (0, 165, 255),
        'frustrado': (0, 0, 255),
        'entendiendo': (255, 255, 0),
        'desconocido': (128, 128, 128)
    }
    
    frame_count = 0
    show_details = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer el frame")
            break
        
        bbox = detector.detect(frame)
        
        if bbox and frame_count % 5 == 0:
            face_crop = detector.crop_face(frame, bbox)
            state, confidence, emotion, all_emotions = classifier.predict(face_crop)
            
            frame_display = detector.draw_bbox(frame, bbox)
            color = state_colors.get(state, (255, 255, 255))
            
            cv2.putText(frame_display, f"Estado: {state.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.putText(frame_display, f"Confianza: {confidence:.0%}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.putText(frame_display, f"Emocion: {emotion}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if show_details and all_emotions:
                y_pos = 135
                for emo, score in all_emotions.items():
                    text = f"{emo}: {score:.1f}%"
                    cv2.putText(frame_display, text, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    y_pos += 20
            
            if frame_count % 30 == 0:
                print(f"[PREDICCION] Estado: {state.upper()} | Confianza: {confidence:.0%} | Emoción: {emotion}")
                if show_details and all_emotions:
                    print(f"  Detalles: {all_emotions}")
        elif bbox:
            frame_display = detector.draw_bbox(frame, bbox)
        else:
            frame_display = frame
            cv2.putText(frame_display, "Sin rostro detectado", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Analisis Cognitivo - DeepFace', frame_display)
        
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_details = not show_details
            print(f"[INFO] Detalles de emociones: {'activados' if show_details else 'desactivados'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Sistema detenido")


if __name__ == "__main__":
    main()
