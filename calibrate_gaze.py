import cv2
from face_detector import FaceDetector
from gaze_detector_debug import GazeDetector


def main():
    print("="*60)
    print("HERRAMIENTA DE CALIBRACION DE GAZE")
    print("="*60)
    print("\nInstrucciones:")
    print("1. Posicionate frente a la camara (60-80cm)")
    print("2. Iluminacion frontal uniforme")
    print("3. Presiona 'c' para iniciar calibracion")
    print("4. Mira directamente a la camara durante 3 segundos")
    print("5. Presiona 'q' para salir")
    print("="*60)
    
    face_detector = FaceDetector()
    gaze_detector = GazeDetector(debug=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la camara")
        return
    
    calibrating = False
    calibration_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks = face_detector.detect(frame)
        
        if landmarks:
            bbox = face_detector.get_bbox_from_landmarks(landmarks, frame.shape)
            frame_display = face_detector.draw_bbox(frame, bbox)
            
            looking, left_ratio, right_ratio = gaze_detector.is_looking_at_camera(landmarks, frame.shape)
            avg_ratio = (left_ratio + right_ratio) / 2.0
            
            status_color = (0, 255, 0) if looking else (0, 0, 255)
            status_text = "MIRANDO" if looking else "NO MIRANDO"
            
            cv2.putText(frame_display, f"Estado: {status_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame_display, f"Left: {left_ratio:.2f}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_display, f"Right: {right_ratio:.2f}", (10, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_display, f"Avg: {avg_ratio:.2f}", (10, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_display, f"Threshold: {gaze_detector.looking_threshold:.1f}", (10, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if calibrating:
                calibration_frames.append((landmarks, frame.shape))
                progress = len(calibration_frames)
                cv2.putText(frame_display, f"CALIBRANDO: {progress}/90", (10, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                if len(calibration_frames) >= 90:
                    print("\n[CALIBRACION] Procesando datos...")
                    ratios = []
                    for lm, shape in calibration_frames:
                        _, l, r = gaze_detector.is_looking_at_camera(lm, shape)
                        ratios.append((l + r) / 2.0)
                    
                    max_ratio = max(ratios)
                    avg_ratio_cal = sum(ratios) / len(ratios)
                    recommended = max_ratio * 1.3
                    
                    print(f"\n[RESULTADO]")
                    print(f"  Ratio promedio: {avg_ratio_cal:.2f}")
                    print(f"  Ratio maximo: {max_ratio:.2f}")
                    print(f"  Threshold actual: {gaze_detector.looking_threshold:.1f}")
                    print(f"  Threshold recomendado: {recommended:.1f}")
                    print(f"\nEn gaze_detector.py linea 12, cambiar a:")
                    print(f"  self.looking_threshold = {recommended:.1f}")
                    
                    calibrating = False
                    calibration_frames = []
            else:
                cv2.putText(frame_display, "Presiona 'c' para calibrar", (10, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            frame_display = frame
            cv2.putText(frame_display, "Sin rostro detectado", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow('Calibracion Gaze', frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and landmarks and not calibrating:
            print("\n[CALIBRACION] Iniciando... Mira la camara directamente!")
            calibrating = True
            calibration_frames = []
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
