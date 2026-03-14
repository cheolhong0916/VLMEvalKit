SUPERCLRVER_sub_shape = {
    "car": ["suv", "wagon", "minivan", "sedan", "truck", "addi", "car"],
    "bus": ["articulated", "regular", "double", "school", "bus"],
    "motorbike": ["chopper", "dirtbike", "scooter", "cruiser", "motorbike"],
    "aeroplane": ["jet", "fighter", "biplane", "airliner", "aeroplane"],
    "bicycle": ["road", "utility", "mountain", "tandem", "bicycle"],
}

inverse_shape = {}
for key, value in SUPERCLRVER_sub_shape.items():
    for v in value:
        inverse_shape[v] = key


class Spatial457_utils:
    def __init__(self):

        return

    def get_random_answer(self, gt):
        import random

        all_attributes = {
            "size": ["small", "large"],
            "shape": [
                "airliner",
                "dirtbike",
                "road bike",
                "tandem bike",
                "suv",
                "wagon",
                "scooter",
                "mountain bike",
                "minivan",
                "sedan",
                "school bus",
                "fighter",
                "chopper",
                "double bus",
                "truck",
                "articulated bus",
                "cruiser",
                "jet",
                "utility bike",
                "regular bus",
                "biplane",
            ],
            "color": [
                "gray",
                "blue",
                "purple",
                "brown",
                "green",
                "cyan",
                "red",
                "yellow",
            ],
            "direction": ["left", "right", "front", "back"],
        }

        gt = gt.lower()
        if gt in ["yes", "no"]:
            return random.choice(["yes", "no"])
        if gt in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return str(random.randint(0, 9))
        for key, value in all_attributes.items():
            if gt in value:
                return random.choice(value)

    def all_answers(self):
        all_attributes = {
            "size": ["small", "large"],
            "shape": [
                "airliner",
                "dirtbike",
                "road bike",
                "tandem bike",
                "suv",
                "wagon",
                "scooter",
                "mountain bike",
                "minivan",
                "sedan",
                "school bus",
                "fighter",
                "chopper",
                "double bus",
                "truck",
                "articulated bus",
                "cruiser",
                "jet",
                "utility bike",
                "regular bus",
                "biplane",
            ],
            "color": [
                "gray",
                "blue",
                "purple",
                "brown",
                "green",
                "cyan",
                "red",
                "yellow",
            ],
            "direction": ["left", "right", "front", "back"],
        }

        all_answers = ""
        for key, value in all_attributes.items():
            captical_value = [x.capitalize() for x in value]
            all_answers += ", ".join(captical_value) + ", "
        return all_answers.strip(", ")

    def is_correct(self, answer, predict):
        text2num = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        predict = str(predict)
        answer = str(answer)

        if predict.lower() == "none":
            predict = "no"

        if predict.lower() == answer.lower():
            return True
        if predict == "0" and answer == "No":
            return True
        if predict.lower() in text2num and text2num[predict.lower()] == answer:
            return True
        if answer.lower() == "yes" and predict in [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ]:
            return True

        if self.category_correct(predict, answer):
            return True
        return False

    def category_correct(self, answer, gt_answer):
        answer = str(answer).lower().split(" ")[0]
        gt_answer = str(gt_answer).lower().split(" ")[0]

        if (
            answer in inverse_shape
            and gt_answer in inverse_shape
            and inverse_shape[answer] == inverse_shape[gt_answer]
        ):
            return True

        return False


class Spatial457_simple_utils(Spatial457_utils):
    """Extended utils with question type detection for simplified instructions."""

    def __init__(self):
        super().__init__()

        self.size_options = ["Small", "Large"]
        self.shape_options = [
            "Airliner", "Dirtbike", "Road bike", "Tandem bike", "Suv", "Wagon",
            "Scooter", "Mountain bike", "Minivan", "Sedan", "School bus", "Fighter",
            "Chopper", "Double bus", "Truck", "Articulated bus", "Cruiser", "Jet",
            "Utility bike", "Regular bus", "Biplane"
        ]
        self.color_options = ["Gray", "Blue", "Purple", "Brown", "Green", "Cyan", "Red", "Yellow"]
        self.direction_options = ["Left", "Right", "Front", "Back"]

    def detect_question_type(self, question):
        """
        Detect the expected answer type from question.
        Returns: 'color', 'shape', 'size', 'direction', 'count', 'yesno'
        """
        q_lower = question.lower()

        # Count questions
        if "how many" in q_lower or "what is the number of" in q_lower:
            return "count"

        # Yes/No questions
        yesno_patterns = [
            "is there", "are there", "does the", "do the",
            "is its shape the same", "is its color the same",
            "is it the same", "are there fewer", "are there more",
            "is the same shape", "is the same color", "have the same"
        ]
        for pattern in yesno_patterns:
            if pattern in q_lower:
                return "yesno"

        # Color questions
        color_patterns = ["what color", "has what color", "is what color", "what is the color"]
        for pattern in color_patterns:
            if pattern in q_lower:
                return "color"

        # Shape questions
        shape_patterns = ["what shape", "is what shape", "shape is it", "what is the shape"]
        for pattern in shape_patterns:
            if pattern in q_lower:
                return "shape"

        # Size questions
        size_patterns = ["what size", "is what size", "how big", "what is the size"]
        for pattern in size_patterns:
            if pattern in q_lower:
                return "size"

        # Direction questions
        direction_patterns = [
            "which direction", "what direction", "facing", "which way",
            "is facing", "faces"
        ]
        for pattern in direction_patterns:
            if pattern in q_lower:
                return "direction"

        # Default: try to infer from question ending
        if q_lower.rstrip("?").endswith("color"):
            return "color"
        if q_lower.rstrip("?").endswith("shape"):
            return "shape"

        return "generic"

    def get_answer_hint(self, q_type):
        """Return appropriate answer hint based on question type."""

        if q_type == "count":
            return "Answer with an integer (0-10)."

        if q_type == "yesno":
            return "Answer with Yes or No."

        if q_type == "color":
            return f"Answer with one of: {', '.join(self.color_options)}."

        if q_type == "shape":
            return f"Answer with one of: {', '.join(self.shape_options)}."

        if q_type == "size":
            return f"Answer with one of: {', '.join(self.size_options)}."

        if q_type == "direction":
            return f"Answer with one of: {', '.join(self.direction_options)}."

        # Generic fallback (still shorter than original)
        return (
            f"Answer with: a color ({', '.join(self.color_options)}), "
            f"a shape (e.g., Sedan, Fighter, Scooter), "
            f"a size (Small/Large), a direction (Left/Right/Front/Back), "
            f"an integer (0-10), or Yes/No."
        )