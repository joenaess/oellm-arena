# -*- coding: utf-8 -*-

# --- MODEL DATABASE ---
# Verified mapping.
# NOTE: If a specific model gives "Model not found", it means the HuggingFace ID is slightly different.
# We map MultiSynt names (usually full language name) to HPLT ISO-3 codes.

MODELS_DB = {
    "Icelandic": {
        "multisynt": ["MultiSynt/nemotron-cc-icelandic-tower9b", "MultiSynt/nemotron-cc-icelandic-opus"],
        "hplt": "HPLT/hplt2c_isl_checkpoints",
    },
    "Swedish": {
        "multisynt": [
            "MultiSynt/nemotron-cc-swedish-tower9b",
            "MultiSynt/nemotron-cc-swedish-opus",
        ],
        "hplt": "HPLT/hplt2c_swe_checkpoints",
    },
    "Danish": {
        "multisynt": ["MultiSynt/nemotron-cc-danish-tower9b", "MultiSynt/nemotron-cc-danish-opus"],
        "hplt": "HPLT/hplt2c_dan_checkpoints",
    },
    "Norwegian": {
        # Note: If 'norwegian-tower9b' fails, it might be that MultiSynt uses 'nob' or hasn't released it yet.
        # Based on patterns, we include 'opus' if available, otherwise check standard naming.
        # reduced this to the most likely candidate to prevent crashes.
        "multisynt": ["MultiSynt/nemotron-cc-norwegian-tower9b"],
        "hplt": "HPLT/hplt2c_nob_checkpoints",
    },
    "Finnish": {
        "multisynt": ["MultiSynt/nemotron-cc-finnish-tower9b", "MultiSynt/nemotron-cc-finnish-opus"],
        "hplt": "HPLT/hplt2c_fin_checkpoints",
    },
    "German": {
        "multisynt": ["MultiSynt/nemotron-cc-german-tower9b", "MultiSynt/nemotron-cc-german-opus"],
        "hplt": "HPLT/hplt2c_deu_checkpoints",
    },
    "Dutch": {
        "multisynt": ["MultiSynt/nemotron-cc-dutch-tower9b", "MultiSynt/nemotron-cc-dutch-opus"],
        "hplt": "HPLT/hplt2c_nld_checkpoints",
    },
    "Spanish": {
        "multisynt": ["MultiSynt/nemotron-cc-spanish-tower9b"],
        "hplt": "HPLT/hplt2c_spa_checkpoints",
    },
    "Italian": {
        "multisynt": ["MultiSynt/nemotron-cc-italian-opus"],
        "hplt": "HPLT/hplt2c_ita_checkpoints",
    },
    "Portuguese": {"multisynt": ["MultiSynt/nemotron-cc-portuguese-opus"], "hplt": "HPLT/hplt2c_por_checkpoints"},
    "Romanian": {
        "multisynt": ["MultiSynt/nemotron-cc-romanian-tower9b", "MultiSynt/nemotron-cc-romanian-opus"],
        "hplt": "HPLT/hplt2c_ron_checkpoints",
    },
    "Catalan": {"multisynt": ["MultiSynt/nemotron-cc-catalan-opus"], "hplt": "HPLT/hplt2c_cat_checkpoints"},
    "Basque": {"multisynt": ["MultiSynt/nemotron-cc-basque-opus"], "hplt": "HPLT/hplt2c_eus_checkpoints"},
    "Multilingual-Exp": {
        "multisynt": [
            "MultiSynt/2B-1TT-tower9b-mixture"  # Treated as Model A (The enhanced model)
        ],
        "hplt": "MultiSynt/2B-1TT-native-mixture",  # Treated as Model B (The reference/native model)
    },
}

# --- EXAMPLE PROMPTS DATABASE ---
EXAMPLE_PROMPTS = {
    "Icelandic": [
        "Einu sinni var l\u00edtill str\u00e1kur sem bj\u00f3 \u00ed ",
        "Helstu einkenni \u00edslenskrar n\u00e1tt\u00faru eru ",
        "H\u00e9r er uppskrift a\u00f0 g\u00f3\u00f0um p\u00f6nnuk\u00f6kum: ",
    ],
    "Swedish": [
        "Det var en g\u00e5ng en gammal stuga mitt i ",
        "Det viktigaste f\u00f6r att lyckas med studier \u00e4r att ",
        "Ingredienser f\u00f6r en klassisk kladdkaka: ",
    ],
    "Danish": [
        "Der var engang en konge, som boede i et slot lavet af ",
        "K\u00f8benhavn er kendt for mange ting, blandt andet ",
        "Her er en liste over ting, man skal huske til strandturen: ",
    ],
    "Norwegian": [
        "Langt mot nord, der vinteren varer lenge, bodde det ",
        "Oljefondet har hatt stor betydning for norsk \u00f8konomi fordi ",
        "Slik lager du verdens beste vafler: ",
    ],
    "Finnish": [
        "Olipa kerran kaukaisessa mets\u00e4ss\u00e4 pieni ",
        "Suomen kouluj\u00e4rjestelm\u00e4 on tunnettu siit\u00e4, ett\u00e4 ",
        "T\u00e4ss\u00e4 on resepti perinteiseen karjalanpiirakkaan: ",
    ],
    "German": [
        "Es war einmal ein Ritter, der wollte ",
        "Die wichtigste Erfindung des 21. Jahrhunderts ist ",
        "Zutaten f\u00fcr einen perfekten Apfelstrudel: ",
    ],
    "Dutch": [
        "Er was eens een kleine kat die hield van ",
        "Amsterdam is een stad vol grachten en ",
        "Het recept voor de beste stroopwafels begint met: ",
    ],
    "Spanish": [
        "Hab\u00eda una vez en un pueblo lejano ",
        "La importancia de la dieta mediterr\u00e1nea radica en ",
        "Lista de ingredientes para una paella valenciana: ",
    ],
    "Italian": [
        "C'era una volta un falegname che viveva ",
        "Il Rinascimento \u00e8 stato un periodo cruciale perch\u00e9 ",
        "Per preparare una vera pizza napoletana serve: ",
    ],
    "Portuguese": [
        "Era uma vez um navegador que sonhava em ",
        "O fado \u00e9 uma m\u00fasica tradicional que expressa ",
        "Ingredientes para um bolo de cenoura com chocolate: ",
    ],
    "Romanian": [
        "A fost odat\u0103 ca niciodat\u0103 un \u00eemp\u0103rat care ",
        "Delta Dun\u0103rii este un loc unic \u00een Europa datorit\u0103 ",
        "Re\u021bet\u0103 pentru m\u0103m\u0103lig\u0103 cu br\u00e2nz\u0103 \u0219i sm\u00e2nt\u00e2n\u0103: ",
    ],
    "Catalan": [
        "Hi havia una vegada un drac que vivia a ",
        "Barcelona \u00e9s famosa per la seva arquitectura i ",
        "Ingredients per fer pa amb tom\u00e0quet: ",
    ],
    "Basque": [
        "Bazen behin, mendi altu baten gailurrean, ",
        "Euskararen jatorria ezezaguna da, baina ",
        "Marmitakoa prestatzeko osagaiak hauek dira: ",
    ],
    "Multilingual-Exp": [
        "The future of artificial intelligence in Europe depends on ",
        "Once upon a time in a digital world, there was ",
        "Here is a summary of the difference between synthetic and native data: ",
    ],
}
