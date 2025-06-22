import vtk
import os
import logging
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("viewer.log", mode='a')
    ]
)


class VTKModelViewer:
    def __init__(self):
        self.actor = None
        self.text_actor = vtk.vtkTextActor()
        self.current_label = ""

        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(800, 800)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        self.render_window.SetWindowName("3D Pose Viewer")

        try:
            self.render_window.SetPosition(500, 200)
        except Exception as e:
            print(f"[poll_commands ERROR] {e}")
            traceback.print_exc()

        # Setup label text actor
        self.text_actor.GetTextProperty().SetFontSize(24)
        self.text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        self.text_actor.SetPosition(10, 740)  # Top-left corner
        self.renderer.AddActor2D(self.text_actor)

        # Setup timer for auto-rotation
        self.rotate_angle = 0
        self.interactor.AddObserver('TimerEvent', self.rotate_callback)
        self.timer_id = None

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            logging.warning(f"Model not found: {model_path}")
            return

        if self.actor:
            self.renderer.RemoveActor(self.actor)

        reader = vtk.vtkOBJReader()
        reader.SetFileName(model_path)
        reader.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.renderer.AddActor(self.actor)

        self.renderer.ResetCamera()
        self.render_window.Render()

        # Extract label and update text
        label = os.path.splitext(os.path.basename(model_path))[0].capitalize()
        self.current_label = label
        self.text_actor.SetInput(f"Pose: {label}")
        logging.info(f"Loaded model: {model_path}")

    def rotate_callback(self, obj, event):
        if self.actor:
            self.rotate_angle = (self.rotate_angle + 1) % 360
            self.actor.SetOrientation(0, self.rotate_angle, 0)
            self.render_window.Render()

    def start(self):
        self.interactor.Initialize()
        self.timer_id = self.interactor.CreateRepeatingTimer(30)  # 30ms ~ 33 FPS
        self.interactor.Start()
