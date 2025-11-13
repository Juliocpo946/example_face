import cv2
from face_detector import FaceDetector
from fer_classifier import FERCognitiveClassifier


def main():
    print("[INFO] Inicializando sistema de análisis cognitivo con FER...")
    
    detector = FaceDetector()
    classifier = FERCognitiveClassifier()
    
    if not classifier.model_loaded:
        print("[ERROR] No se pudo cargar el modelo. Ejecuta:")
        print("  pip install -r requirements_fer.txt")
        return
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la cámara")
        return
    
    print("[INFO] Sistema iniciado correctamente")
    print("[INFO] Presiona 'q' para salir")
    
    state_colors = {
        'concentrado': (0, 255, 0),
        'distraido': (0, 165, 255),
        'frustrado': (0, 0, 255),
        'entendiendo': (255, 255, 0),
        'desconocido': (128, 128, 128)
    }
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer el frame")
            break
        
        bbox = detector.detect(frame)
        
        if bbox:
            face_crop = detector.crop_face(frame, bbox)
            state, confidence, emotion = classifier.predict(face_crop)
            
            frame_display = detector.draw_bbox(frame, bbox)
            color = state_colors.get(state, (255, 255, 255))
            
            cv2.putText(frame_display, f"Estado: {state.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.putText(frame_display, f"Confianza: {confidence:.0%}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.putText(frame_display, f"Emocion: {emotion}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if frame_count % 15 == 0:
                print(f"[PREDICCION] Estado: {state.upper()} | Confianza: {confidence:.0%} | Emoción: {emotion}")
        else:
            frame_display = frame
            cv2.putText(frame_display, "Sin rostro detectado", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Analisis Cognitivo - FER CNN', frame_display)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Sistema detenido")


if __name__ == "__main__":
    main()