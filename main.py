import cv2
from face_detector import FaceDetector
from emotion_classifier import EmotionClassifier
from gaze_detector import GazeDetector
from drowsiness_detector import DrowsinessDetector


def main():
    print("="*60)
    print("SISTEMA DE ANALISIS COGNITIVO")
    print("MediaPipe + HSEmotion + Gaze + Drowsiness")
    print("="*60)
    print("[INFO] Inicializando componentes...")
    
    face_detector = FaceDetector()
    emotion_classifier = EmotionClassifier()
    gaze_detector = GazeDetector()
    drowsiness_detector = DrowsinessDetector()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la camara")
        return
    
    print("[INFO] Sistema iniciado")
    print("="*60)
    print("CONTROLES:")
    print("  'q' - Salir")
    print("  'd' - Detalles emociones")
    print("  'i' - Info sistema")
    print("  'r' - Reset somnolencia")
    print("="*60)
    
    state_colors = {
        'concentrado': (0, 255, 0),
        'distraido': (0, 165, 255),
        'frustrado': (0, 0, 255),
        'entendiendo': (255, 255, 0),
        'somnoliento': (147, 20, 255),
        'desconocido': (128, 128, 128)
    }
    
    frame_count = 0
    show_details = False
    show_info = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks = face_detector.detect(frame)
        
        if landmarks and frame_count % 3 == 0:
            bbox = face_detector.get_bbox_from_landmarks(landmarks, frame.shape)
            face_crop = face_detector.crop_face(frame, bbox)
            
            state, confidence, emotion, all_emotions = emotion_classifier.predict(face_crop)
            looking_at_camera, gaze_direction = gaze_detector.analyze_gaze(landmarks, frame.shape)
            is_drowsy, drowsy_level, drowsy_stats = drowsiness_detector.detect(landmarks, frame.shape)
            
            if is_drowsy or drowsy_level >= 2:
                state = 'somnoliento'
                confidence = 1.0 - drowsy_stats['ear']
            elif not looking_at_camera and state != 'frustrado':
                state = 'distraido'
            
            color = state_colors.get(state, (255, 255, 255))
            frame_display = face_detector.draw_bbox(frame, bbox, color)
            
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
                cv2.putText(frame_display, f"EAR: {drowsy_stats['ear']:.2f}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 25
                cv2.putText(frame_display, f"MAR: {drowsy_stats['mar']:.2f}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 25
                cv2.putText(frame_display, f"Parpadeos: {drowsy_stats['blinks']}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 25
                cv2.putText(frame_display, f"Bostezos: {drowsy_stats['yawns']}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 25
                cv2.putText(frame_display, f"Nivel: {drowsy_level}/3", (10, y_offset),
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
                print(f"[ESTADO] {state.upper()} | Confianza: {confidence:.0%} | "
                      f"Emocion: {emotion} | Mirando: {gaze_status} | "
                      f"Somnolencia: {drowsy_level}/3 | EAR: {drowsy_stats['ear']:.2f} | "
                      f"Parpadeos: {drowsy_stats['recent_blinks']}")
        
        elif landmarks:
            bbox = face_detector.get_bbox_from_landmarks(landmarks, frame.shape)
            frame_display = face_detector.draw_bbox(frame, bbox)
        else:
            frame_display = frame
            cv2.putText(frame_display, "Sin rostro detectado", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow('Sistema Cognitivo Completo', frame_display)
        
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_details = not show_details
            print(f"[INFO] Detalles: {'ON' if show_details else 'OFF'}")
        elif key == ord('i'):
            show_info = not show_info
            print(f"[INFO] Info sistema: {'ON' if show_info else 'OFF'}")
        elif key == ord('r'):
            drowsiness_detector.reset_drowsy_state()
            print("[INFO] Estado de somnolencia reiniciado")
    
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Sistema detenido")


if __name__ == "__main__":
    main()