"""
Hand-labeled few-shot examples used in the labeler prompt.

We include 2 of the 4 hand-labeled cases. The other 2 (D2N002, D2N004) are
held out so we can spot-check the labeler's outputs against them and estimate
labeling noise. Do NOT use held-out examples here.

These examples are taken directly from labels_clean.jsonl and must match
that file's labeling conventions exactly. They are excluded from the
evaluation set automatically via FEW_SHOT_IDS to prevent data leakage.
"""

# D2N001 - Martha Collins (annual exam, multiple chronic conditions, no
# single presenting complaint -> duration=null)
EXAMPLE_1_ENCOUNTER_ID = "D2N001"
EXAMPLE_1_DIALOGUE = """[doctor] hi , martha . how are you ?
[patient] i'm doing okay . how are you ?
[doctor] martha is a 50-year-old female with a past medical history significant for congestive heart failure , depression and hypertension who presents for her annual exam . so , martha , it's been a year since i've seen you . how are you doing ?
[patient] i'm doing well . i've been traveling a lot recently .
[doctor] how are you doing watching your diet ? i know we've talked about watching a low sodium diet .
[patient] i've been doing well with that .
[doctor] and any symptoms like chest pains , shortness of breath , any swelling in your legs ?
[patient] no , not that i've noticed .
[doctor] and then in terms of your depression , how are you doing ?
[patient] therapy has been helping a lot . i've been going every week for the past year .
[doctor] no feelings of wanting to harm yourself or hurt others ?
[patient] no , nothing like that .
[doctor] and then in terms of your high blood pressure , how are you doing remembering your medications ?
[patient] i'm still forgetting to take my blood pressure medication .
[doctor] i know that you were endorsing some nasal congestion from some of the fall pollen and allergies . any other symptoms , nausea or vomiting , abdominal pain , anything like that ?
[patient] no , nothing like that .
[doctor] for your first problem your congestive heart failure , i wan na continue you on your current medications . but i do wan na increase your lisinopril to 40 milligrams a day . i also wan na start you on some lasix , 20 milligrams a day . and have you continue to watch your diet . i also wan na repeat another echocardiogram .
[doctor] from a depression standpoint , i do n't feel the need to start you on any medications this year .
[doctor] and then for your last problem your hypertension , i'd like to see you take the lisinopril as directed . i want you to record your blood pressures every day for like a week . and for your annual exam , you're due for a mammogram .
[patient] okay ."""


EXAMPLE_1_LABEL = {
    "name": "Martha",
    "age": 50,
    "symptoms": ["nasal congestion"],
    "duration": None,
    "negated_symptoms": [
        "chest pain",
        "shortness of breath",
        "leg swelling",
        "suicidal ideation",
        "homicidal ideation",
        "nausea",
        "vomiting",
        "abdominal pain",
    ],
    "history": ["congestive heart failure", "depression", "hypertension"],
    "diagnosis": ["congestive heart failure"],
    "treatment": [
        {"type": "medication", "detail": "increase lisinopril to 40 mg daily"},
        {"type": "medication", "detail": "start lasix 20 mg daily"},
        {"type": "medication", "detail": "continue current chf medications"},
        {"type": "test", "detail": "repeat echocardiogram"},
        {"type": "test", "detail": "repeat lipid panel"},
        {"type": "test", "detail": "mammogram"},
        {"type": "counseling", "detail": "continue low-sodium diet"},
        {"type": "counseling",
         "detail": "take lisinopril as directed; record home bp daily for one week"},
    ],
}


# D2N003 - John Perry (acute presenting complaint with clear duration,
# multiple symptoms, hedged endorsement of hematuria)
EXAMPLE_2_ENCOUNTER_ID = "D2N003"
EXAMPLE_2_DIALOGUE = """[doctor] hi , john . how are you ?
[doctor] so john is a 61-year-old male with a past medical history significant for kidney stones , migraines and reflux , who presents with some back pain . so john , what's going on with your back ?
[patient] uh , i'm feeling a lot of the same pain that i had when i had kidney stones about two years ago , so i'm a little concerned . it started from the right side and kinda moved over , and now i feel it in the left side of my back .
[doctor] and how many days has this been going on for ?
[patient] the last four days .
[doctor] do you have any blood in your urine ?
[patient] um , uh , i think i do . it's kind of hard to detect , but it does look a little off-color .
[doctor] have you had any other symptoms like nausea and vomiting ?
[patient] if i'm exerting myself , like climbing stairs or running to catch the bus , i feel a little dizzy and a little light headed , and i still feel a little bit more pain in my abdomen .
[doctor] let's talk about your migraines . i know we started you on the imitrex .
[patient] i've been pretty diligent about taking the meds .
[doctor] and how about your acid reflux ?
[patient] i've been pretty good with the diet .
[doctor] any other symptoms ? muscle aches , chest pain , body aches , anything like that ?
[patient] i have some body aches because i think i'm favoring my back when i'm walking .
[doctor] for your first problem , your back pain , i think you're having a recurrence of your kidney stones . so i wan na go ahead and order a ct scan without contrast of your abdomen and pelvis . i'm also gon na order you some ultram 50 milligrams as needed every six hours for pain . and i want you to push fluids and strain your urine .
[doctor] for your migraines , let's continue you on the imitrex . and for your reflux , we have you on the protonix 40 milligrams a day . do you need a refill ?
[patient] actually , i do need a refill .
[doctor] if your symptoms worsen , just give me a call ."""

EXAMPLE_2_LABEL = {
    "name": "John",
    "age": 61,
    "symptoms": [
        "back pain",
        "hematuria",
        "dizziness",
        "lightheadedness",
        "abdominal pain",
        "body aches",
    ],
    "duration": "4 days",
    "negated_symptoms": [],
    "history": ["kidney stones", "migraines", "gastroesophageal reflux"],
    "diagnosis": [
        "recurrent kidney stones",
    ],
    "treatment": [
        {"type": "medication",
         "detail": "ultram 50 mg every 6 hours as needed for pain"},
        {"type": "medication", "detail": "continue imitrex"},
        {"type": "medication", "detail": "refill protonix 40 mg daily"},
        {"type": "test", "detail": "ct abdomen and pelvis without contrast"},
        {"type": "counseling", "detail": "push fluids and strain urine"},
        {"type": "follow_up", "detail": "call if symptoms worsen"},
    ],
}


FEW_SHOT_EXAMPLES = [
    {
        "encounter_id": EXAMPLE_1_ENCOUNTER_ID,
        "dialogue": EXAMPLE_1_DIALOGUE,
        "label": EXAMPLE_1_LABEL,
    },
    {
        "encounter_id": EXAMPLE_2_ENCOUNTER_ID,
        "dialogue": EXAMPLE_2_DIALOGUE,
        "label": EXAMPLE_2_LABEL,
    },
]


# Encounter IDs to exclude from the evaluation set, since these dialogues
# appear in the prompt and would constitute data leakage if also evaluated.
FEW_SHOT_IDS = {ex["encounter_id"] for ex in FEW_SHOT_EXAMPLES}
