"""Dietary Reference Intakes (DRI) calculator: BMI, EER, macros, vitamins and minerals by age and sex.
Based on the USDA DRI calculator: https://www.nal.usda.gov/human-nutrition-and-food-safety/dri-calculator"""
 
from __future__ import annotations
from dataclasses import dataclass
 
 
# Physical activity coefficients for the EER equations
PA_MALE = {
    "Sedentary":   1.00,
    "Low Active":  1.11,
    "Active":      1.25,
    "Very Active": 1.48,
}
PA_FEMALE = {
    "Sedentary":   1.00,
    "Low Active":  1.12,
    "Active":      1.27,
    "Very Active": 1.45,
}
 
# Pregnancy / lactation calorie additions (kcal/day)
PREGNANCY_ADDITIONS = {
    "1st trimester": 0,
    "2nd trimester": 340,
    "3rd trimester": 452,
}
LACTATION_ADDITIONS = {
    "0-6 months":  330,
    "7-12 months": 400,
}
 
# DRI lookup tables (RDA or AI, and UL), keyed by (sex, age_min, age_max); age_max is inclusive.
# ND = not determinable
 
# Fibre (g/day), AI
FIBRE_TABLE = {
    ("male",   19, 30): 38,
    ("male",   31, 50): 38,
    ("male",   51, 70): 30,
    ("male",   71, 120):30,
    ("female", 19, 30): 25,
    ("female", 31, 50): 25,
    ("female", 51, 70): 21,
    ("female", 71, 120):21,
}
 
# Water (L/day), AI
WATER_TABLE = {
    ("male",   19, 30): (3.7, "about 16 cups"),
    ("male",   31, 50): (3.7, "about 16 cups"),
    ("male",   51, 70): (3.7, "about 16 cups"),
    ("male",   71, 120):(3.7, "about 16 cups"),
    ("female", 19, 30): (2.7, "about 11 cups"),
    ("female", 31, 50): (2.7, "about 11 cups"),
    ("female", 51, 70): (2.7, "about 11 cups"),
    ("female", 71, 120):(2.7, "about 11 cups"),
}
 
# Vitamins: {name: {(sex, age_min, age_max): (rda_or_ai, ul)}}, ul = "ND" where not determinable
VITAMIN_TABLE = {
    "Vitamin A (mcg)": {
        ("male",   19, 30): (900,  3000),
        ("male",   31, 50): (900,  3000),
        ("male",   51, 70): (900,  3000),
        ("male",   71, 120):(900,  3000),
        ("female", 19, 30): (700,  3000),
        ("female", 31, 50): (700,  3000),
        ("female", 51, 70): (700,  3000),
        ("female", 71, 120):(700,  3000),
    },
    "Vitamin C (mg)": {
        ("male",   19, 30): (90,   2000),
        ("male",   31, 50): (90,   2000),
        ("male",   51, 70): (90,   2000),
        ("male",   71, 120):(90,   2000),
        ("female", 19, 30): (75,   2000),
        ("female", 31, 50): (75,   2000),
        ("female", 51, 70): (75,   2000),
        ("female", 71, 120):(75,   2000),
    },
    "Vitamin D (mcg)": {
        ("male",   19, 30): (15,   100),
        ("male",   31, 50): (15,   100),
        ("male",   51, 70): (15,   100),
        ("male",   71, 120):(20,   100),
        ("female", 19, 30): (15,   100),
        ("female", 31, 50): (15,   100),
        ("female", 51, 70): (15,   100),
        ("female", 71, 120):(20,   100),
    },
    "Vitamin E (mg)": {
        ("male",   19, 120):(15,   1000),
        ("female", 19, 120):(15,   1000),
    },
    "Vitamin K (mcg)": {
        ("male",   19, 30): (120,  "ND"),
        ("male",   31, 50): (120,  "ND"),
        ("male",   51, 70): (120,  "ND"),
        ("male",   71, 120):(120,  "ND"),
        ("female", 19, 30): (90,   "ND"),
        ("female", 31, 50): (90,   "ND"),
        ("female", 51, 70): (90,   "ND"),
        ("female", 71, 120):(90,   "ND"),
    },
    "Vitamin B6 (mg)": {
        ("male",   19, 50): (1.3,  100),
        ("male",   51, 70): (1.7,  100),
        ("male",   71, 120):(1.7,  100),
        ("female", 19, 50): (1.3,  100),
        ("female", 51, 70): (1.5,  100),
        ("female", 71, 120):(1.5,  100),
    },
    "Vitamin B12 (mcg)": {
        ("male",   19, 120):(2.4,  "ND"),
        ("female", 19, 120):(2.4,  "ND"),
    },
    "Thiamin (mg)": {
        ("male",   19, 120):(1.2,  "ND"),
        ("female", 19, 120):(1.1,  "ND"),
    },
    "Riboflavin (mg)": {
        ("male",   19, 120):(1.3,  "ND"),
        ("female", 19, 120):(1.1,  "ND"),
    },
    "Niacin (mg)": {
        ("male",   19, 120):(16,   35),
        ("female", 19, 120):(14,   35),
    },
    "Folate (mcg)": {
        ("male",   19, 120):(400,  1000),
        ("female", 19, 120):(400,  1000),
    },
    "Choline (g)": {
        ("male",   19, 120):(0.55, 3.5),
        ("female", 19, 120):(0.425,3.5),
    },
    "Pantothenic Acid (mg)": {
        ("male",   19, 120):(5,    "ND"),
        ("female", 19, 120):(5,    "ND"),
    },
    "Biotin (mcg)": {
        ("male",   19, 120):(30,   "ND"),
        ("female", 19, 120):(30,   "ND"),
    },
}
 
MINERAL_TABLE = {
    "Calcium (mg)": {
        ("male",   19, 70): (1000, 2500),
        ("male",   71, 120):(1200, 2000),
        ("female", 19, 50): (1000, 2500),
        ("female", 51, 120):(1200, 2000),
    },
    "Chloride (g)": {
        ("male",   19, 50): (2.3,  3.6),
        ("male",   51, 70): (2.0,  3.6),
        ("male",   71, 120):(1.8,  3.6),
        ("female", 19, 50): (2.3,  3.6),
        ("female", 51, 70): (2.0,  3.6),
        ("female", 71, 120):(1.8,  3.6),
    },
    "Chromium (mcg)": {
        ("male",   19, 50): (35,   "ND"),
        ("male",   51, 120):(30,   "ND"),
        ("female", 19, 50): (25,   "ND"),
        ("female", 51, 120):(20,   "ND"),
    },
    "Copper (mcg)": {
        ("male",   19, 120):(900,  10000),
        ("female", 19, 120):(900,  10000),
    },
    "Fluoride (mg)": {
        ("male",   19, 120):(4,    10),
        ("female", 19, 120):(3,    10),
    },
    "Iodine (mcg)": {
        ("male",   19, 120):(150,  1100),
        ("female", 19, 120):(150,  1100),
    },
    "Iron (mg)": {
        ("male",   19, 120):(8,    45),
        ("female", 19, 50): (18,   45),
        ("female", 51, 120):(8,    45),
    },
    "Magnesium (mg)": {
        ("male",   19, 30): (400,  350),
        ("male",   31, 120):(420,  350),
        ("female", 19, 30): (310,  350),
        ("female", 31, 120):(320,  350),
    },
    "Manganese (mg)": {
        ("male",   19, 120):(2.3,  11),
        ("female", 19, 120):(1.8,  11),
    },
    "Molybdenum (mcg)": {
        ("male",   19, 120):(45,   2000),
        ("female", 19, 120):(45,   2000),
    },
    "Phosphorus (g)": {
        ("male",   19, 70): (0.7,  4.0),
        ("male",   71, 120):(0.7,  3.0),
        ("female", 19, 70): (0.7,  4.0),
        ("female", 71, 120):(0.7,  3.0),
    },
    "Potassium (mg)": {
        ("male",   19, 30): (3400, "ND"),
        ("male",   31, 50): (3400, "ND"),
        ("male",   51, 120):(3400, "ND"),
        ("female", 19, 30): (2600, "ND"),
        ("female", 31, 50): (2600, "ND"),
        ("female", 51, 120):(2600, "ND"),
    },
    "Selenium (mcg)": {
        ("male",   19, 120):(55,   400),
        ("female", 19, 120):(55,   400),
    },
    "Sodium (mg)": {
        ("male",   19, 50): (1500, 2300),
        ("male",   51, 70): (1300, 2300),
        ("male",   71, 120):(1200, 2300),
        ("female", 19, 50): (1500, 2300),
        ("female", 51, 70): (1300, 2300),
        ("female", 71, 120):(1200, 2300),
    },
    "Zinc (mg)": {
        ("male",   19, 120):(11,   40),
        ("female", 19, 120):(8,    40),
    },
}
 
 
def _lookup(table: dict, sex: str, age: int):
    """Find the value in a (sex, age_min, age_max) keyed table."""
    for (s, a_min, a_max), val in table.items():
        if s == sex and a_min <= age <= a_max:
            return val
    return None
 
 
def calculate_dri(
    sex: str,           # "male" | "female"
    age: int,           # years
    weight_kg: float,
    height_cm: float,
    activity: str,      # "Sedentary" | "Low Active" | "Active" | "Very Active"
    pregnancy: str = "None",   # "None" | "1st trimester" | "2nd trimester" | "3rd trimester"
    lactation: str = "None",   # "None" | "0-6 months" | "7-12 months"
) -> dict:
    """Compute a full DRI profile matching the USDA DRI calculator.
    Returns a dict with bmi, eer, macros, vitamins and minerals."""
    h_m = height_cm / 100.0
 
    bmi = round(weight_kg / (h_m ** 2), 1)
    if   bmi < 18.5: bmi_class = "Underweight"
    elif bmi < 25.0: bmi_class = "Normal weight"
    elif bmi < 30.0: bmi_class = "Overweight"
    else:            bmi_class = "Obese"
 
    # EER (kcal/day): National Academies DRI for Energy equations, adults 19+
    pa_table = PA_MALE if sex == "male" else PA_FEMALE
    pa = pa_table.get(activity, 1.00)
 
    if sex == "male":
        eer = 662 - (9.53 * age) + pa * ((15.91 * weight_kg) + (539.6 * h_m))
    else:
        eer = 354 - (6.91 * age) + pa * ((9.36 * weight_kg) + (726 * h_m))
 
    extra = 0
    if pregnancy != "None":
        extra += PREGNANCY_ADDITIONS.get(pregnancy, 0)
    if lactation != "None":
        extra += LACTATION_ADDITIONS.get(lactation, 0)
 
    eer = round(eer + extra)
 
    # AMDR: carbs 45-65%, fat 20-35% of energy; 4 kcal/g carbs, 9 kcal/g fat
    carb_min_g  = round(eer * 0.45 / 4)
    carb_max_g  = round(eer * 0.65 / 4)
    protein_rda = round(0.8 * weight_kg)
    fat_min_g   = round(eer * 0.20 / 9)
    fat_max_g   = round(eer * 0.35 / 9)
 
    fibre = _lookup(FIBRE_TABLE, sex, age)
    water_entry = _lookup(WATER_TABLE, sex, age)
    water_l, water_cups = water_entry if water_entry else (None, None)
 
    macros = {
        "Carbohydrate":     {"value": f"{carb_min_g}–{carb_max_g} g",   "note": "45–65% of EER"},
        "Total Fiber":      {"value": f"{fibre} g" if fibre else "—",    "note": "AI"},
        "Protein":          {"value": f"{protein_rda} g",                "note": "RDA (0.8 g/kg)"},
        "Fat":              {"value": f"{fat_min_g}–{fat_max_g} g",      "note": "20–35% of EER"},
        "Total Water":      {"value": f"{water_l} L ({water_cups})" if water_l else "—", "note": "AI"},
        "α-Linolenic Acid": {"value": "1.6 g" if sex == "male" else "1.1 g", "note": "AI (Omega-3)"},
        "Linoleic Acid":    {"value": "17 g" if sex == "male" else "12 g",   "note": "AI (Omega-6)"},
    }
 
    vitamins = {}
    for name, age_map in VITAMIN_TABLE.items():
        val = _lookup(age_map, sex, age)
        if val:
            rda, ul = val
            vitamins[name] = {"rda": rda, "ul": ul}
 
    minerals = {}
    for name, age_map in MINERAL_TABLE.items():
        val = _lookup(age_map, sex, age)
        if val:
            rda, ul = val
            minerals[name] = {"rda": rda, "ul": ul}
 
    return {
        "inputs": {
            "sex": sex, "age": age, "weight_kg": weight_kg,
            "height_cm": height_cm, "activity": activity,
        },
        "bmi":      bmi,
        "bmi_class": bmi_class,
        "eer":      eer,
        "macros":   macros,
        "vitamins": vitamins,
        "minerals": minerals,
        "_raw": {
            "carb_min_g": carb_min_g, "carb_max_g": carb_max_g,
            "protein_g":  protein_rda,
            "fat_min_g":  fat_min_g,  "fat_max_g":  fat_max_g,
            "calories":   eer,
            # Sodium UL from the mineral table (mg), converted to g
            "sodium_g":   round((minerals.get("Sodium (mg)", {}).get("ul") or 2300) / 1000, 3),
            # WHO: free sugars <10% of energy, <5% for added benefit (4 kcal/g)
            "sugar_limit_g":      round(eer * 0.10 / 4, 1),
            "sugar_limit_low_g":  round(eer * 0.05 / 4, 1),
        },
    }