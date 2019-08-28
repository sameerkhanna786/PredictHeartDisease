from django.forms import Form
from django.forms import Select
from django.forms import IntegerField
from django.forms import NullBooleanField
from django.forms import FloatField
from django.forms import TypedChoiceField

SEX_CHOICES = (
	('', '-------'),
    (0, "Female"),
    (1, "Male"),
)

CP_CHOICES = (
	('', '-----------------'),
    (3, "Typical Angina"),
    (2, "Atypical Angina"),
    (1, "Non-anginal pain"),
    (0, "Asymtomatic"),
)

FBS_CHOICES = (
	('', '----'),
    (0, "No"),
    (1, "Yes"),
)

EXANG_CHOICES = (
	('', '----'),
    (0, "No"),
    (1, "Yes"),
)

RESTECG_CHOICES = (
	('', '---------------------------------------------'),
    (0, "Normal"),
    (1, "Having ST-T wave abnormality"),
    (2, "Definite/probable left ventricular hypertrophy"),
)

SLOPE_CHOICES = (
	('', '-------------'),
    (0, "Upsloping"),
    (1, "Flat"),
    (2, "Downsloping"),
)

CA_CHOICES = (
	('', '--'),
    (0, "0"),
    (1, "1"),
    (2, "2"),
    (3, "3"),
)

THAL_CHOICES = (
	('', '------------------'),
    (0, "Normal"),
    (1, "Fixed Defect"),
    (2, "Reversible Defect"),
)

class predForm(Form):
	age = IntegerField(label="What is your age?", max_value = 100, min_value = 1, required=False)
	sex = TypedChoiceField(label="What is your sex?", coerce = int, choices=SEX_CHOICES, required=False, empty_value = None)
	cp = TypedChoiceField(label="What is your chest pain type?", coerce = int, choices=CP_CHOICES, required=False, empty_value = None)
	trestbps = IntegerField(label="What is your resting blood pressure in mm Hg?", required=False)
	chol = IntegerField(label="What is your serum cholestoral in mg/dl?", required=False)
	fbs = TypedChoiceField(label="Is your fasting blood sugar > 120 mg/dl?", coerce = int, choices=FBS_CHOICES, required=False, empty_value = None)
	restecg = TypedChoiceField(label="How is your resting ecg?", coerce = int, choices=RESTECG_CHOICES, required=False, empty_value = None)
	thalach = IntegerField(label="What is your maximum heart rate achieved during excercise?", required=False)
	exang = TypedChoiceField(label="Do you have exercise induced angina?", coerce = int, choices=EXANG_CHOICES, required=False, empty_value = None)
	oldpeak = FloatField(label="What is your ST depression induced by exercise relative to rest?", required=False)
	slope = TypedChoiceField(label="What is the slope of your peak excercise ST segment?", coerce = int, choices=SLOPE_CHOICES, required=False, empty_value = None)
	ca = TypedChoiceField(label="What is the number of your major blood vessels colored during flourosopy?", coerce = int, choices=CA_CHOICES, required=False, empty_value = None)
	thal = TypedChoiceField(label="What are your thal results?", coerce = int, choices=THAL_CHOICES, required=False, empty_value = None)




