from typing import Dict, Any, Optional

DISEASE_METADATA = {
    "blast": {
        "scientific_name": "Magnaporthe oryzae (Foliar/Neck Blast)",
        "pathogen_type": "Fungal Ascomycete",
        "primary_trigger": "Prolonged leaf wetness (>10 hours), relative humidity >85%, temperatures between 20°C–28°C, and excessive nitrogenous fertilizer.",
        "why_it_occurred": (
            "Fungal conidia of Magnaporthe oryzae germinate and penetrate leaf epidermal tissue when leaves remain wet from high humidity or morning dew. "
            "Under moderate temperatures with low vapor pressure deficit (VPD), fungal appressoria exert mechanical and enzymatic turgor pressure to breach host cells. "
            "Excessive nitrogen creates tender, succulent foliage that accelerates lesion expansion into typical diamond/spindle-shaped necrotic spots with gray centers."
        ),
        "chemical_cure": (
            "1. Spray Tricyclazole 75% WP @ 0.6 g/L of water at the first appearance of leaf lesions.\n"
            "2. Alternatively, spray Isoprothiolane 40% EC @ 1.5 mL/L or Kasugamycin 3% SL @ 2.5 mL/L.\n"
            "3. Ensure thorough coverage of both upper and lower leaf surfaces during early morning or late afternoon."
        ),
        "biological_control": (
            "1. Foliar spray of Pseudomonas fluorescens (commercial talc formulation) @ 2.5 kg/ha or 5 g/L.\n"
            "2. Apply Trichoderma harzianum or Bacillus subtilis to enhance systemic acquired resistance (SAR)."
        ),
        "preventive_cultural_practices": (
            "1. Immediately suspend top-dressing of urea/nitrogen until lesions turn brown and stop expanding.\n"
            "2. Apply potassium (MOP) to strengthen cell walls against fungal penetration.\n"
            "3. Maintain proper field spacing to ensure adequate air circulation and promote rapid leaf drying."
        )
    },
    "bacterial_leaf_blight": {
        "scientific_name": "Xanthomonas oryzae pv. oryzae (BLB)",
        "pathogen_type": "Bacterial Pathogen",
        "primary_trigger": "High temperatures (25°C–34°C), high humidity (>75%), unseasonal rain, and wind damage creating microscopic wounds.",
        "why_it_occurred": (
            "Xanthomonas bacteria enter through natural leaf openings (hydathodes at leaf margins) or wounds caused by wind friction, rain splash, and handling. "
            "The bacteria multiply within the xylem vessels, producing extracellular polysaccharides that clog water transport, causing water-soaked wavy lesions "
            "that turn straw-yellow and dry outward from the leaf tip downward."
        ),
        "chemical_cure": (
            "1. Spray Streptocycline @ 1 g + Copper Oxychloride (COC) 50% WP @ 25 g dissolved in 10 liters of water.\n"
            "2. Repeat spray after 10–12 days if disease progression persists.\n"
            "3. Avoid spraying during heavy winds or wet foliage to prevent mechanical dissemination."
        ),
        "biological_control": (
            "1. Foliar application of plant extract (Neem oil @ 3% or NSKE @ 5%).\n"
            "2. Seedling root dip or foliar spray with Pseudomonas fluorescens @ 10 g/L."
        ),
        "preventive_cultural_practices": (
            "1. Drain stagnant water from the field for 3–4 days to reduce canopy humidity.\n"
            "2. Strictly avoid clipping seedling leaf tips during transplanting.\n"
            "3. Apply balanced N:P:K in split doses (avoid single heavy urea doses); add extra potassium (10–15 kg/ha)."
        )
    },
    "bacterial_leaf_streak": {
        "scientific_name": "Xanthomonas oryzae pv. oryzicola (BLS)",
        "pathogen_type": "Bacterial Pathogen",
        "primary_trigger": "Warm, overcast, humid days with intermittent drizzle and mechanical wind abrasion.",
        "why_it_occurred": (
            "Bacteria enter via stomata and multiply in parenchymatous leaf tissue between veins, restricted by parallel vascular bundles to form narrow, "
            "translucent interveinal streaks. Under high morning humidity, bacterial amber-colored droplets (ooze) form on the leaf surface and are splashed "
            "to adjacent healthy leaves by rain droplets."
        ),
        "chemical_cure": (
            "1. Spray Copper Hydroxide 77% WP @ 2.0 g/L or Copper Oxychloride @ 2.5 g/L.\n"
            "2. Combine with Plantomycin or Streptomycin sulfate (100–150 ppm) in severe field outbreaks."
        ),
        "biological_control": (
            "1. Foliar spray of bio-agent Pseudomonas fluorescens @ 5 g/L.\n"
            "2. Spray bio-formulations with silicon or silicate foliar nutrients to harden epidermal tissues."
        ),
        "preventive_cultural_practices": (
            "1. Avoid working or traversing the field when foliage is wet with morning dew.\n"
            "2. Disinfect farm tools between plots with a 1% sodium hypochlorite solution.\n"
            "3. Eradicate grassy weed hosts (e.g., Leersia hexandra) along field bunds."
        )
    },
    "bacterial_panicle_blight": {
        "scientific_name": "Burkholderia glumae (BPB)",
        "pathogen_type": "Bacterial Panicle Pathogen",
        "primary_trigger": "High minimum nighttime temperatures (>28°C) and hot daytime temperatures (>32°C) during flowering/heading (65–85 days).",
        "why_it_occurred": (
            "Burkholderia glumae survives epiphytically on rice foliage during vegetative growth. As the crop transitions into reproductive heading (age >65 days), "
            "high ambient temperatures trigger the bacterial toxoflavin virulence gene. The pathogen colonizes emerging florets, causing grain abortion, "
            "spikelet discoloration, and empty upright panicles."
        ),
        "chemical_cure": (
            "1. Spray Kasugamycin 3% SL @ 2.0 mL/L or Oxolinic acid at early booting and panicle emergence.\n"
            "2. Copper-based sprays (Copper hydroxide @ 2 g/L) help suppress secondary bacterial load on foliage."
        ),
        "biological_control": (
            "1. Seed treatment and boot-stage spray with antagonistic Bacillus amyloliquefaciens strains.\n"
            "2. Apply biocontrol seed coating before nursery sowing."
        ),
        "preventive_cultural_practices": (
            "1. Avoid late nitrogen top-dressing during heading, which prolongs susceptibility.\n"
            "2. Use certified pathogen-free certified seed lots.\n"
            "3. Adjust sowing windows so peak flowering does not coincide with peak summer temperatures."
        )
    },
    "brown_spot": {
        "scientific_name": "Bipolaris oryzae / Cochliobolus miyabeanus",
        "pathogen_type": "Fungal Deuteromycete",
        "primary_trigger": "Soil nutrient imbalance (potassium, silicon, or zinc deficiency), poor drainage, or unseasonal moisture stress.",
        "why_it_occurred": (
            "Brown spot is historically recognized as a 'poor farmer's disease' because it thrives when the host plant is physiologically weakened by nutrient "
            "deficiency (especially potash and micronutrients) or root asphyxiation in poorly aerated soil. Air-borne conidia land on stressed leaves, "
            "producing circular-to-oval brown lesions with yellow chlorotic halos."
        ),
        "chemical_cure": (
            "1. Spray Propiconazole 25% EC @ 1.0 mL/L or Mancozeb 75% WP @ 2.0 g/L.\n"
            "2. In severe cases, spray Hexaconazole 5% EC @ 2.0 mL/L or Azoxystrobin + Difenoconazole @ 1.0 mL/L."
        ),
        "biological_control": (
            "1. Seed treatment with Trichoderma viride @ 4 g/kg seed.\n"
            "2. Foliar application of cow-urine fermented herbal decoction (Panchagavya @ 3%)."
        ),
        "preventive_cultural_practices": (
            "1. Soil application of Zinc Sulfate @ 25 kg/ha and Potassium (MOP) @ 40–50 kg/ha in splits.\n"
            "2. Ensure intermittent wetting and drying (AWD) rather than continuous stagnation to oxygenate roots.\n"
            "3. Treat seeds with Carbendazim (2 g/kg seed) before sowing next season."
        )
    },
    "dead_heart": {
        "scientific_name": "Scirpophaga incertulas (Yellow Stem Borer)",
        "pathogen_type": "Lepidopteran Insect Pest",
        "primary_trigger": "Vegetative crop stage (30–60 days), dense tillering, cloudy warm weather, and presence of adult female moths.",
        "why_it_occurred": (
            "Adult female yellow stem borer moths deposit egg masses covered in buff-colored hairs near leaf tips. The newly hatched larvae bore into the tiller stem base, "
            "feeding on the inner vascular tissues. Severing the vascular bundles halts water and nutrient flow to the central growing spindle, causing it to wither, "
            "turn straw-colored, and pull out easily (the characteristic 'dead heart' symptom)."
        ),
        "chemical_cure": (
            "1. Broadcast Cartap Hydrochloride 4G @ 10 kg/acre or Fipronil 0.3% G @ 10 kg/acre with sand.\n"
            "2. Alternatively, spray Chlorantraniliprole 18.5% SC @ 0.3 mL/L or Flubendiamide 39.35% SC @ 0.2 mL/L.\n"
            "3. Maintain 2–3 cm standing water in the field during granular insecticide application for 48 hours."
        ),
        "biological_control": (
            "1. Install pheromone traps (Scirpo-lure) @ 5–8 traps/acre for pest monitoring and mass trapping.\n"
            "2. Release egg parasitoid Trichogramma japonicum @ 100,000 adults/ha (5 times at weekly intervals starting 30 days after transplanting)."
        ),
        "preventive_cultural_practices": (
            "1. Clip leaf tips before transplanting to destroy stem borer egg masses.\n"
            "2. Avoid excessive nitrogen fertilizer, which produces succulent thick tillers favored by boring larvae.\n"
            "3. Harvest at ground level to remove stubble harboring over-wintering / diapausing pupae."
        )
    },
    "downy_mildew": {
        "scientific_name": "Sclerophthora macrospora (Downy Mildew / Crazy Top)",
        "pathogen_type": "Oomycete Pathogen",
        "primary_trigger": "Submerged seedlings, stagnant nursery water, cool temperatures (18°C–24°C), and waterlogged soils.",
        "why_it_occurred": (
            "Water-borne biflagellate zoospores of Sclerophthora infect young seedlings during prolonged submergence. The pathogen spreads systemically inside tillers, "
            "causing foliar yellowing, twisted spiral leaf growth, downy white fungal fuzz under high humidity, and severe stunting."
        ),
        "chemical_cure": (
            "1. Spray Metalaxyl 8% + Mancozeb 64% WP (Ridomil MZ) @ 2.5 g/L.\n"
            "2. Alternatively, spray Cymoxanil 8% + Mancozeb 64% WP @ 2.0 g/L at early symptom detection."
        ),
        "biological_control": (
            "1. Seed bio-priming with Trichoderma asperellum @ 10 g/kg.\n"
            "2. Soil application of neem cake @ 200 kg/ha to reduce soil-borne oospore viability."
        ),
        "preventive_cultural_practices": (
            "1. Improve nursery and mainfield drainage; avoid deep stagnant standing water on young tillers.\n"
            "2. Rogue and destroy severely distorted 'crazy top' hills to prevent oospore replenishment in soil.\n"
            "3. Eradicate collateral grassy weeds (Digitaria, Echinochloa) surrounding nursery canals."
        )
    },
    "hispa": {
        "scientific_name": "Dicladispa armigera (Rice Hispa)",
        "pathogen_type": "Chrysomelid Insect Pest",
        "primary_trigger": "Dense planting, heavy vegetative tillering stage, high humidity, and overcast skies.",
        "why_it_occurred": (
            "Both adult spiny black beetles and internal leaf-mining grubs attack the crop. Adults scrape the upper green leaf surface leaving characteristic parallel white "
            "chlorotic streaks. Grubs burrow between the upper and lower epidermal layers, creating irregular transparent blister patches that cause the whole leaf to wither."
        ),
        "chemical_cure": (
            "1. Spray Chlorpyrifos 20% EC @ 2.5 mL/L or Quinalphos 25% EC @ 2.0 mL/L.\n"
            "2. In case of severe leaf miner infestation, spray Cartap Hydrochloride 50% SP @ 2.0 g/L."
        ),
        "biological_control": (
            "1. Spray Neem Seed Kernel Extract (NSKE 5%) or Azadirachtin 10,000 ppm @ 2.0 mL/L.\n"
            "2. Encourage native predators such as spiders (Oxyopes, Lycosa) and larval parasitoids (Eulophidae)."
        ),
        "preventive_cultural_practices": (
            "1. Sweep-netting: Pass an insect hand net across the canopy during early morning to capture and destroy adult beetles.\n"
            "2. Clip and destroy top 2–3 inches of leaf tips before transplanting.\n"
            "3. Avoid staggered planting in adjacent plots to break the pest generation cycle."
        )
    },
    "tungro": {
        "scientific_name": "Rice Tungro Spherical & Bacilliform Viruses (RTSV & RTBV)",
        "pathogen_type": "Viral Complex (Vector Transmitted)",
        "primary_trigger": "High populations of Green Leafhoppers (Nephotettix virescens) and overlapping staggered planting.",
        "why_it_occurred": (
            "Tungro is a dual-virus infection transmitted semi-persistently by green leafhoppers feeding on leaf phloem. The virus disrupts chlorophyll biosynthesis and root nutrient uptake, "
            "manifesting as intense yellow-to-orange foliar discoloration starting from leaf tips, severe plant stunting, reduced tillering, and delayed panicle emergence."
        ),
        "chemical_cure": (
            "1. Note: Direct anti-viral chemicals do not exist; intervention MUST target the leafhopper vector.\n"
            "2. Spray Imidacloprid 17.8% SL @ 0.3 mL/L or Thiamethoxam 25% WG @ 0.2 g/L.\n"
            "3. Alternatively, spray Dinotefuran 20% SG @ 0.4 g/L or Pymetrozine 50% WG @ 0.6 g/L."
        ),
        "biological_control": (
            "1. Spray neem oil (3 mL/L) with liquid soap (1 mL/L) to repel leafhopper feeding.\n"
            "2. Conserve green mirid bugs (Cyrtorhinus lividipennis) which feed on leafhopper egg masses."
        ),
        "preventive_cultural_practices": (
            "1. Rogue and bury infected yellow-stunted hills immediately to eliminate viral inoculation reservoirs.\n"
            "2. Observe a mandatory crop-free fallow period (at least 21 days) between consecutive rice seasons.\n"
            "3. Plant resistant or tolerant cultivars (e.g., IR64, Vikramarya)."
        )
    },
    "normal": {
        "scientific_name": "Oryza sativa (Healthy Foliage)",
        "pathogen_type": "Non-Pathological / Normal Vigorous Growth",
        "primary_trigger": "Balanced soil nutrition, healthy root aeration, and weather conditions within normal physiological thresholds.",
        "why_it_occurred": (
            "The crop canopy exhibits vibrant, unblemished green coloration, indicating active chlorophyll synthesis, balanced nitrogen-potassium metabolism, "
            "and absence of invasive pathogen colonization or insect trauma."
        ),
        "chemical_cure": (
            "No chemical pesticide or fungicide intervention is required. Avoid prophylactic chemical spraying to conserve natural beneficial predators (spiders, parasitoids)."
        ),
        "biological_control": (
            "Maintain soil biological health by applying Trichoderma-enriched compost or farmyard manure (FYM) during field preparation."
        ),
        "preventive_cultural_practices": (
            "1. Practice Alternate Wetting and Drying (AWD) water management to promote robust root architecture and save irrigation water.\n"
            "2. Continue regular field scouting twice a week to detect any early pest or disease foci.\n"
            "3. Apply recommended top-dressing fertilizers based on leaf color chart (LCC) recommendations."
        )
    }
}


def build_detailed_advisory_report(
    disease_key: str,
    confidence: float,
    variety: str,
    age: int,
    weather_data: Optional[Dict[str, Any]] = None,
    location_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Synthesize deep learning predictions, agronomic metadata, and live
    meteorological context into a comprehensive causal explanation and action plan.
    """
    info = DISEASE_METADATA.get(disease_key, DISEASE_METADATA["normal"])
    weather = weather_data or {}
    location = location_data or {}

    temp = weather.get("temperature", 32.5)
    humidity = weather.get("humidity", 45)
    rain = weather.get("rain", 0.0)
    vpd = weather.get("vpd", 1.2)
    wind = weather.get("wind_speed", 4.0)
    city = location.get("city", "Hyderabad")
    region = location.get("region", "Telangana")

    # 1. Calculate Combined Epidemiological Risk Percentage
    # Formula combines model confidence (60%) with weather environmental pressure (40%)
    weather_risk_factor = 0.3
    if disease_key in ["blast", "downy_mildew"]:
        if humidity >= 80: weather_risk_factor = 0.95
        elif humidity >= 65: weather_risk_factor = 0.65
        else: weather_risk_factor = 0.25
    elif disease_key in ["bacterial_leaf_blight", "bacterial_leaf_streak"]:
        if rain > 0 or humidity >= 75: weather_risk_factor = 0.90
        elif temp >= 28: weather_risk_factor = 0.60
        else: weather_risk_factor = 0.30
    elif disease_key == "bacterial_panicle_blight":
        if temp >= 32 and age >= 65: weather_risk_factor = 0.95
        elif age >= 65: weather_risk_factor = 0.60
        else: weather_risk_factor = 0.20
    elif disease_key in ["dead_heart", "hispa"]:
        if 35 <= age <= 70: weather_risk_factor = 0.80
        else: weather_risk_factor = 0.40

    combined_risk = round((confidence * 0.60 + weather_risk_factor * 0.40) * 100, 1)
    if disease_key == "normal":
        combined_risk = round((1.0 - confidence) * 20, 1)  # Low risk for healthy

    if combined_risk >= 75:
        risk_category = "CRITICAL"
        risk_color = "#ef4444"
    elif combined_risk >= 50:
        risk_category = "HIGH"
        risk_color = "#f97316"
    elif combined_risk >= 30:
        risk_category = "MODERATE"
        risk_color = "#eab308"
    else:
        risk_category = "LOW"
        risk_color = "#10b981"

    # 2. Build Multi-Factor Contextual Causality Explanation ("Why it occurred")
    weather_synthesis = (
        f"At your location in {city}, {region} (Lat {location.get('latitude', 17.38)}°N, Lon {location.get('longitude', 78.48)}°E), "
        f"the current ambient temperature is {temp}°C with {humidity}% relative humidity, {rain} mm rainfall, and {wind} km/h wind. "
    )
    if humidity >= 80:
        weather_synthesis += f"The atmospheric relative humidity ({humidity}%) is exceptionally high with low vapor pressure deficit ({vpd} kPa), creating extended periods of free surface moisture on the canopy that promotes pathogen sporulation and cellular infection. "
    elif temp >= 32:
        weather_synthesis += f"Elevated ambient temperatures ({temp}°C) combined with intense solar radiation create heat stress on reproductive tillers, predisposing spikelets to bacterial colonization. "
    else:
        weather_synthesis += f"Current microclimate indicators show mild atmospheric evaporative demand (VPD: {vpd} kPa) and moderate thermal kinetics. "

    phenology_synthesis = (
        f"For cultivar '{variety}' at chronological crop age of {age} days (phenological stage: {_get_stage_description(age)}), "
        f"tissue susceptibility is heightened to {_get_stage_vulnerability(age, disease_key)}."
    )

    full_causality = f"{weather_synthesis}\n\n{phenology_synthesis}\n\nBiological Mechanism: {info['why_it_occurred']}"

    return {
        "scientific_name": info["scientific_name"],
        "pathogen_type": info["pathogen_type"],
        "primary_trigger": info["primary_trigger"],
        "risk_percentage": combined_risk,
        "risk_category": risk_category,
        "risk_color": risk_color,
        "location_summary": f"{city}, {region}",
        "weather_summary": f"{temp}°C • {humidity}% RH • Wind {wind} km/h",
        "why_it_occurred": full_causality,
        "action_plan": {
            "chemical_cure": info["chemical_cure"],
            "biological_control": info["biological_control"],
            "cultural_prevention": info["preventive_cultural_practices"]
        }
    }


def _get_stage_description(age: int) -> str:
    if age < 50: return "Vegetative Tillering"
    if age < 62: return "Panicle Primordium Initiation"
    if age < 72: return "Heading & Anthesis (Flowering)"
    if age < 82: return "Grain Filling & Milk Stage"
    return "Maturity & Ripening"


def _get_stage_vulnerability(age: int, disease: str) -> str:
    if age >= 65 and disease == "bacterial_panicle_blight":
        return "panicle blast and bacterial panicle blight due to open blooming spikelets"
    elif age < 55 and disease == "dead_heart":
        return "yellow stem borer penetration due to soft, rapidly expanding vegetative tillers"
    elif disease == "blast":
        return "foliar blast lesions across actively transpiring younger leaf canopies"
    elif disease == "bacterial_leaf_blight":
        return "vascular bacterial multiplication along expanding leaf margins"
    return "opportunistic pathological infection"
