import cv2
from face_detector import FaceDetector
from emotion_classifier import EmotionClassifier


def main():
    print("="*60)
    print("SISTEMA DE ANALISIS DE EMOCIONES")
    print("HSEmotion - EfficientNet-B0")
    print("="*60)
    print("[INFO] Inicializando componentes...")
    
    face_detector = FaceDetector()
    emotion_classifier = EmotionClassifier()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la camara")
        return
    
    print("[INFO] Sistema iniciado")
    print("="*60)
    print("CONTROLES:")
    print("  'q' - Salir")
    print("  'd' - Detalles emociones")
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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        bbox = face_detector.detect(frame)
        
        if bbox and frame_count % 3 == 0:
            face_crop = face_detector.crop_face(frame, bbox)
            state, confidence, emotion, all_emotions = emotion_classifier.predict(face_crop)
            
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
                print(f"[ESTADO] {state.upper()} | Confianza: {confidence:.0%} | Emocion: {emotion}")
        
        elif bbox:
            frame_display = face_detector.draw_bbox(frame, bbox)
        else:
            frame_display = frame
            cv2.putText(frame_display, "Sin rostro detectado", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow('Analisis de Emociones', frame_display)
        
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_details = not show_details
            print(f"[INFO] Detalles: {'ON' if show_details else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Sistema detenido")


if __name__ == "__main__":
    main()
