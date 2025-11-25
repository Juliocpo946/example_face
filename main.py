import cv2
import logging
from datetime import datetime
from config import AppConfig
from video_capture import VideoCapture
from analysis_pipeline import AnalysisPipeline
from display_renderer import DisplayRenderer


logging.basicConfig(
    level=logging.ERROR,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("APP")


class Application:
    def __init__(self, config: AppConfig = None):
        self._config = config or AppConfig()
        self._capture = VideoCapture()
        self._pipeline = AnalysisPipeline(self._config)
        self._renderer = DisplayRenderer(self._config.display)
        self._frame_count = 0
        self._running = False

    def run(self):
        print("=" * 60)
        print("SISTEMA DE ANALISIS DE EMOCIONES Y ATENCION")
        print("HSEmotion + MediaPipe")
        print("=" * 60)
        print("[INFO] Inicializando componentes...")
        
        if not self._capture.open():
            logger.error("No se pudo acceder a la camara")
            print("[ERROR] No se pudo acceder a la camara")
            return
        
        width, height = self._capture.frame_size
        self._pipeline.update_image_size(width, height)
        
        print("[INFO] Sistema iniciado")
        print("=" * 60)
        print("CONTROLES:")
        print("  'q' - Salir")
        print("  'd' - Mostrar/ocultar detalles de emociones")
        print("  'r' - Recalibrar posicion de cabeza")
        print("=" * 60)
        
        self._running = True
        last_log_state = None
        
        while self._running:
            ret, frame = self._capture.read()
            if not ret:
                logger.error("Error al leer frame de la camara")
                break
            
            if self._frame_count % self._config.process_every_n_frames == 0:
                state, bbox = self._pipeline.process(frame)
            else:
                state = self._pipeline.last_state
                bbox = self._pipeline.last_bbox
            
            if state:
                display = self._renderer.render(frame, state, bbox)
                
                if self._frame_count % 30 == 0 and state.face_detected:
                    if state.final_state != last_log_state:
                        print(f"[ESTADO] {state.final_state.upper()} | Emocion: {state.emotion} | Confianza: {state.confidence:.0%}")
                        last_log_state = state.final_state
            else:
                display = frame
            
            cv2.imshow("Analisis de Emociones y Atencion", display)
            
            self._frame_count += 1
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self._running = False
            elif key == ord("d"):
                self._renderer.toggle_details()
                print(f"[INFO] Detalles: {'activados' if self._renderer._show_details else 'desactivados'}")
            elif key == ord("r"):
                self._pipeline.reset_calibration()
                print("[INFO] Recalibrando... mire a la pantalla")
        
        self._shutdown()

    def _shutdown(self):
        print("[INFO] Cerrando sistema...")
        self._capture.release()
        self._pipeline.release()
        cv2.destroyAllWindows()
        print("[INFO] Sistema detenido")


def main():
    config = AppConfig()
    app = Application(config)
    app.run()


if __name__ == "__main__":
    main()