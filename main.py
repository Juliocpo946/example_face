import cv2
from face_detector import FaceDetector
from ensemble_classifier import EnsembleCognitiveClassifier
from gaze_detector import GazeDetector


def main():
    print("="*60)
    print("SISTEMA DE ANÁLISIS COGNITIVO - ENSEMBLE + GAZE")
    print("="*60)
    print("[INFO] Inicializando sistema completo...")
    print("[INFO] Modelos: HSEmotion (50%) + DeepFace (30%) + Py-Feat (20%)")
    print("[INFO] Suavizado temporal: 15 frames")
    print("[INFO] NOTA: Primera ejecución descargará ~600MB de modelos")
    print("="*60)
    
    face_detector = FaceDetector()
    emotion_classifier = EnsembleCognitiveClassifier()
    gaze_detector = GazeDetector()
    
    if not any(emotion_classifier.models_loaded.values()):
        print("[ERROR] No se pudo cargar ningún modelo. Ejecuta:")
        print("  pip install -r requirements.txt")
        return
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la cámara")
        return
    
    print("[INFO] Sistema iniciado correctamente")
    print("="*60)
    print("CONTROLES:")
    print("  'q' - Salir del programa")
    print("  'd' - Mostrar/ocultar detalles de emociones")
    print("  'g' - Activar/desactivar análisis de mirada")
    print("  'i' - Mostrar/ocultar información del sistema")
    print("="*60)
    
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
    show_info = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        bbox = face_detector.detect(frame)
        
        if bbox and frame_count % 5 == 0:
            face_crop = face_detector.crop_face(frame, bbox)
            
            state, confidence, emotion, all_emotions = emotion_classifier.predict(face_crop)
            
            looking_at_camera, gaze_direction, eye_count = gaze_detector.analyze_gaze(face_crop)
            
            if not looking_at_camera and state != 'frustrado':
                state = 'distraido'
            
            if show_gaze:
                frame_display = gaze_detector.draw_eye_analysis(
                    frame, bbox, looking_at_camera, gaze_direction, eye_count
                )
            else:
                frame_display = face_detector.draw_bbox(frame, bbox)
            
            color = state_colors.get(state, (255, 255, 255))
            
            y_offset = 30
            cv2.putText(frame_display, f"Estado: {state.upper()}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            y_offset += 40
            cv2.putText(frame_display, f"Confianza: {confidence:.0%}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            y_offset += 35
            cv2.putText(frame_display, f"Emocion: {emotion}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            y_offset += 35
            gaze_status = "SI" if looking_at_camera else "NO"
            gaze_color = (0, 255, 0) if looking_at_camera else (0, 0, 255)
            cv2.putText(frame_display, f"Mirando: {gaze_status}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, gaze_color, 2)
            
            if show_info:
                y_offset += 35
                models_active = sum(emotion_classifier.models_loaded.values())
                cv2.putText(frame_display, f"Modelos: {models_active}/3", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if show_details and all_emotions:
                y_offset += 30
                cv2.putText(frame_display, "--- Detalles ---", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                y_offset += 25
                
                sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
                for emo, score in sorted_emotions:
                    text = f"{emo}: {score:.1f}%"
                    cv2.putText(frame_display, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                    y_offset += 20
            
            if frame_count % 30 == 0:
                print(f"[ESTADO] {state.upper()} | Confianza: {confidence:.0%} | Emoción: {emotion} | Mirando: {gaze_status}")
        
        elif bbox:
            frame_display = face_detector.draw_bbox(frame, bbox)
        else:
            frame_display = frame
            cv2.putText(frame_display, "Sin rostro detectado", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow('Sistema Cognitivo Ensemble', frame_display)
        
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_details = not show_details
            print(f"[INFO] Detalles: {'ON' if show_details else 'OFF'}")
        elif key == ord('g'):
            show_gaze = not show_gaze
            print(f"[INFO] Análisis de mirada: {'ON' if show_gaze else 'OFF'}")
        elif key == ord('i'):
            show_info = not show_info
            print(f"[INFO] Información sistema: {'ON' if show_info else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("="*60)
    print("[INFO] Sistema detenido")
    print("="*60)


if __name__ == "__main__":
    main()
