"""Hand-authored natural-language probes.

Every probe was built in one direction only: a target product was picked from
the catalog first, and the shopper's words were written afterwards **without
copying the product's own text**. That ordering matters. The official simulator
constructs its constraints *from the target's own `features` and `details`, so
the query and the document share vocabulary by construction. Real shoppers do
not have the product page in front of them.

`tools.stress` measures, per probe, what fraction of the shopper's content words
actually occur in the target's catalog text, and prints it. A probe with high
overlap is an easy probe and should be read as such. That number is the guard
against writing a test that flatters us.

Nothing in the agent was tuned on this set.

Tags
----
natural            plain shopper phrasing
colloquial         regional or informal usage ("trainers", "jumper")
non_catalog        deliberately uses words the catalog does not use
vague              underspecified opener, narrowed later
brand              names a brand, as shoppers often do
typo              misspellings
multi_attr         several constraints in one turn
negation           states what they do *not* want
contradiction      retracts and replaces a constraint
category_switch    abandons the category mid-session
control_easy       expected to be easy; a sanity check on the runner
control_impossible the target has a near-identical twin in the catalog
"""
from __future__ import annotations

# Each probe: target parent_asin, the shopper's turns in order, and tags.
PROBES: list[dict] = [
    {
        "id": "watch_diver",
        "target": "B000GX3IIM",   # Invicta 8929 Pro Diver, gold-tone, black dial
        "turns": [
            "I need a men's dive watch",
            "self-winding, not battery powered",
            "gold coloured with a dark face",
            "about 40mm across",
        ],
        "tags": ["non_catalog", "multi_attr"],
    },
    {
        "id": "watch_ladies_daydate",
        "target": "B09Z31F3PD",   # Citizen Quartz, two-tone steel, day and date
        "turns": [
            "a wristwatch for my wife",
            "metal strap, classic rather than sporty",
            "it should show the day and the date",
            "two tone, silver and gold",
        ],
        "tags": ["natural"],
    },
    {
        "id": "sneaker_gym_mesh",
        "target": "B004ZIP5HQ",   # PUMA Voltaic 3, synthetic and mesh, foam footbed
        "turns": [
            "men's trainers for the gym",
            "mesh upper so my feet breathe",
            "cushioned insole",
            "puma if they do one",
        ],
        "tags": ["colloquial", "brand"],
    },
    {
        "id": "sneaker_denim_slipon",
        "target": "B07TYB7X4R",   # XIANV denim canvas slip-on
        "turns": [
            "casual shoes for women",
            "made of denim",
            "slip on, no laces",
        ],
        "tags": ["natural"],
    },
    {
        "id": "dress_vintage_floral",
        "target": "B0188S239G",   # retro floral vintage dress, 3/4 sleeve, calf length
        "turns": [
            "a vintage style dress",
            "floral print, 1950s look",
            "three quarter sleeves, down to the calf",
            "zips up the back",
        ],
        "tags": ["non_catalog", "multi_attr"],
    },
    {
        "id": "sandal_beach_arch",
        "target": "B079ZN4S63",   # Panama Jack flip flops, EVA sole, arch support
        "turns": [
            "men's flip flops for the beach",
            "fine to get them wet",
            "with arch support",
        ],
        "tags": ["natural"],
    },
    {
        "id": "sandal_toddler_closed",
        "target": "B07MXTVQ89",   # ELAPHURUS boys closed-toe sport sandals
        "turns": [
            "sandals for my toddler son",
            "closed toe so he does not hurt his toes",
            "for outdoor play in summer",
        ],
        "tags": ["natural"],
    },
    {
        "id": "tee_startrek",
        "target": "B077PFMZ5D",   # Popfunk Star Trek uniform tee, 100% cotton
        "turns": [
            "a star trek t shirt",
            "the uniform design",
            "cotton",
        ],
        "tags": ["control_easy", "brand"],
    },
    {
        "id": "sunglasses_fitover",
        "target": "B0BX5GKSXL",   # URUMQI fit-over polarized wrap-around
        "turns": [
            "sunglasses that fit over my prescription glasses",
            "polarised, with uv protection",
            "wrap around style",
        ],
        "tags": ["natural"],
    },
    {
        "id": "sunglasses_gucci",
        "target": "B071D877KZ",   # Gucci GG0141S black square acetate
        "turns": [
            "designer sunglasses",
            "gucci, square shape",
            "black acetate frame",
        ],
        "tags": ["brand"],
    },
    {
        "id": "loafer_leather",
        "target": "B00DNNPDB2",   # Olukai Nalukai, 100% leather
        "turns": [
            "women's leather slip on shoes",
            "real leather, not synthetic",
            "rubber sole",
        ],
        "tags": ["natural", "negation"],
    },
    {
        "id": "slipper_knit_women",
        "target": "B07FNNTR76",   # Snug Leaves knit slippers, memory foam, wool lining
        "turns": [
            "cosy slippers for my wife",
            "knitted, with memory foam",
            "warm lining inside",
            "sturdy enough to step outside in",
        ],
        "tags": ["natural", "multi_attr"],
    },
    {
        "id": "flat_leather_walking",
        "target": "B008OTSTXO",   # Ahnu Karma Flat, 100% leather
        "turns": [
            "women's flat shoes",
            "leather",
            "good for walking",
        ],
        "tags": ["vague"],
    },
    {
        "id": "necklace_butterfly",
        "target": "B098Q55YTZ",   # VIROMY gold butterfly pendant, 17in + extender
        "turns": [
            "a necklace with a butterfly on it",
            "gold plated",
            "about 17 inches, with an extender chain",
        ],
        "tags": ["natural"],
    },
    {
        "id": "necklace_cross",
        "target": "B09SLT2DK5",   # luomart cross necklace for girls
        "turns": [
            "a cross necklace",
            "for a teenage girl",
            "as a religious gift",
        ],
        "tags": ["natural"],
    },
    {
        "id": "slipper_men_scuff",
        "target": "B0912MMSQS",   # ONCAI men's memory foam scuff slippers, knit
        "turns": [
            "house shoes for men",
            "memory foam",
            "knitted upper",
            "sturdy enough to take the bins out in",
        ],
        "tags": ["colloquial"],
    },
    {
        "id": "sweater_men_bomber",
        "target": "B07N3JH2L8",   # Goodthreads merino/acrylic bomber, zipper
        "turns": [
            "a men's jumper",
            "merino wool blend",
            "zip up, bomber style",
        ],
        "tags": ["colloquial", "non_catalog"],
    },
    {
        "id": "sweater_women_merino",
        "target": "B07KTLMH4L",   # Lark & Ro 100% merino v-neck
        "turns": [
            "women's sweater",
            "pure merino wool",
            "v neck, long sleeves",
        ],
        "tags": ["natural"],
    },
    {
        "id": "raincoat_anorak",
        "target": "B07BCP8DG5",   # Rokka&Rolla lightweight hooded anorak
        "turns": [
            "a rain jacket for women",
            "lightweight, with a hood",
            "zip front, longer length",
        ],
        "tags": ["natural"],
    },
    {
        "id": "jogger_champion",
        "target": "B08ZJWTCDR",   # Champion men's joggers, 100% cotton, drawstring
        "turns": [
            "men's joggers",
            "cotton",
            "drawstring waist",
            "champion make",
        ],
        "tags": ["brand"],
    },
    # --- the failure modes the demo transcript exposed ---------------------
    {
        "id": "switch_shoes_to_pants",
        "target": "B09BFDBRNW",   # Yovela women's high-waisted baggy sweatpants
        "turns": [
            "I am looking for shoes",
            "men's black running shoes",
            "actually never mind, I need women's sweatpants",
            "high waisted and baggy",
            "cotton blend",
        ],
        "tags": ["category_switch"],
    },
    {
        "id": "contradiction_leather_canvas",
        "target": "B08B7X2BWJ",   # Amazon Essentials men's canvas lace-up beach sneaker
        "turns": [
            "men's casual shoes",
            "leather ones",
            "actually not leather, canvas is better",
            "lace up, for the beach",
        ],
        "tags": ["contradiction"],
    },
    {
        "id": "negation_turtleneck",
        "target": "B07P2JV6B1",   # Goodthreads merino turtleneck, lightweight
        "turns": [
            "a men's sweater",
            "not a v neck",
            "turtleneck, merino wool, lightweight",
        ],
        "tags": ["negation"],
    },
    {
        "id": "typo_raincoat",
        "target": "B08242X27K",   # 4HOW women's waterproof hooded raincoat
        "turns": [
            "womens rainocat",
            "waterprrof and lightwieght",
            "with a hoood",
        ],
        "tags": ["typo"],
    },
    {
        "id": "vague_clear_backpack",
        "target": "B0BQBWZXPC",   # Vorspack clear backpack, reinforced bottom
        "turns": [
            "I need a bag for school",
            "a see through one",
            "clear backpack with a sturdy bottom",
        ],
        "tags": ["vague"],
    },
    {
        "id": "control_twin_ballet_flat",
        "target": "B0788CLZJF",   # Nova Utopia Mary Jane flats — twin of B0788BDY2K
        "turns": [
            "women's mary jane ballet flats",
            "synthetic material",
            "rubber sole",
        ],
        "tags": ["control_impossible"],
    },
]

# Sent once the shopper's script runs out, so the agent still gets its full ten
# turns. A real shopper who has said everything they know has nothing left to
# add — this is deliberately harsher than the official simulator, which keeps
# feeding fresh constraints lifted from the target for as long as the session
# runs.
FILLER = [
    "that's all I can think of",
    "anything else you can show me?",
    "nothing more to add",
]
