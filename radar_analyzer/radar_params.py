from typing import List
from dataclasses import field

class RadarExperimentParams:
    \"\"\"Class to store parameters for a radar experiment.\"\"\"

    def __init__(self):
        self.circles = []

    def add_circle(self, enabled: bool, angle: float):
        \"\"\"Add a new circle to the experiment.\"\"\"
        self.circles.append({\"enabled\": enabled, \"angle\": angle})

    def update_circle_enabled(self, index: int, enabled: bool):
        \"\"\"Update the enabled status for a specific circle.\"\"\"
        if 0 <= index < len(self.circles):
            self.circles[index][\"enabled\"] = enabled

    def update_circle_angle(self, index: int, angle: float):
        \"\"\"Update the angle for a specific circle.\"\"\"
        if 0 <= index < len(self.circles):
            # Clamp angle to typical radar limits if needed, though UI should enforce
            clamped_angle = max(-90.0, min(90.0, angle))
            self.circles[index][\"angle\"] = clamped_angle
            print(f\"Updated circle {index} angle to: {clamped_angle}\") # Debug print
        else:
            print(f\"Error: Invalid circle index {index} for angle update.\") # Debug print

class ExperimentData:
    \"\"\"Class to store collected data for an experiment.\"\"\"

class SamplingCircle:
    def __init__(self, enabled: bool, distance: float, radius: float, angle: float, color: str, label: str):
        self.enabled = enabled
        self.distance = distance
        self.radius = radius
        self.angle = angle
        self.color = color
        self.label = label

class RadarExperimentParams:
    circles: List[SamplingCircle] = field(default_factory=lambda: [
        SamplingCircle(enabled=True, distance=5.0, radius=0.5, angle=0.0, color="lime", label="Primary"),
        SamplingCircle(enabled=False, distance=5.0, radius=0.5, angle=28.0, color="cyan", label="Left"),
        SamplingCircle(enabled=False, distance=5.0, radius=0.5, angle=-28.0, color="yellow", label="Right"),
    ])

    use_directional_distance: bool = False  # Added parameter