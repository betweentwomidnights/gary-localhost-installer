"""
Configuration file containing all available variations for MelodyFlow transformations.
Each variation includes a descriptive prompt and processing parameters.
"""

VARIATIONS = {
    # Acoustic Instruments
    'accordion_folk': {
        'prompt': "Lively accordion music with a European folk feeling, perfect for a travel documentary about traditional culture and street performances in Paris",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },
    'banjo_bluegrass': {
        'prompt': "Authentic bluegrass banjo band performance with rich picking patterns, ideal for a heartfelt documentary about American rural life and traditional crafts",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },
    'piano_classical': {
        'prompt': "Expressive classical piano performance with dynamic range and emotional depth, ideal for a luxury brand commercial",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },
    'celtic': {
        'prompt': "Traditional Celtic arrangement with fiddle and flute, perfect for a documentary about Ireland's stunning landscapes and ancient traditions",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },
    'strings_quartet': {
        'prompt': "Elegant string quartet arrangement with rich harmonies and expressive dynamics, perfect for wedding ceremony music",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },

    # Synthesizer Variations
    'synth_retro': {
        'prompt': "1980s style synthesizer melody with warm analog pads and arpeggios, perfect for a nostalgic sci-fi movie soundtrack",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },
    'synth_modern': {
        'prompt': "Modern electronic production with crisp digital synthesizer arpeggios and vocoder effects, ideal for a tech product launch video",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },
    'synth_ambient': {
        'prompt': "Atmospheric synthesizer pads with reverb and delay, perfect for a meditation app or wellness commercial",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },
    'synth_edm': {
        'prompt': "High-energy EDM synth saw leads with sidechain compression, pitch bends, perfect for sports highlights or action sequences",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },

    # Band Arrangements
    'rock_band': {
        'prompt': "Full rock band arrangement with electric guitars, bass, and drums, perfect for an action movie trailer",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },

    # Hybrid/Special
    'cinematic_epic': {
        'prompt': "Epic orchestral arrangement with modern hybrid elements, synthesizers, and percussion, perfect for movie trailers",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },
    'lofi_chill': {
        'prompt': "Lo-fi hip hop style with vinyl crackle, mellow piano, and tape saturation, perfect for study or focus playlists",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },
    'synth_bass': {
        'prompt': "Deep analog synthesizer bassline with modern production and subtle modulation, perfect for electronic music production",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },

    'retro_rpg': {
        'prompt': "16-bit era JRPG soundtrack with bright melodic synthesizers, orchestral elements, and adventurous themes, perfect for a fantasy video game battle scene or overworld exploration",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },

    'steel_drums': {
        'prompt': "Vibrant Caribbean steel drum ensemble with tropical percussion and uplifting melodies, perfect for a beach resort commercial or travel documentary",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },
    'chiptune': {
        'prompt': "8-bit video game soundtrack with arpeggiated melodies and classic NES-style square waves, perfect for a retro platformer or action game",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },
    'gamelan_fusion': {
        'prompt': "Indonesian gamelan ensemble with metallic percussion, gongs, and ethereal textures, perfect for a meditation app or spiritual documentary",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },

    'music_box': {
        'prompt': "Delicate music box melody with gentle bell tones and ethereal ambiance, perfect for a children's lullaby or magical fantasy scene",
        'default_flowstep': 0.12,  # Changed from 'flowstep' to 'default_flowstep'
        'steps': 25
    },

    # Hip Hop / Trap Percussion
    'trap_808': {
        'prompt': "Modern trap beat with booming 808 bass, crisp hi-hat rolls, and punchy snares, perfect for a contemporary hip hop music video with dramatic slow motion scenes",
        'default_flowstep': 0.12,
        'steps': 25
    },
    'lo_fi_drums': {
        'prompt': "Vinyl-processed lo-fi hip hop drums with warm tape saturation, subtle sidechain compression, and occasional vinyl crackle, ideal for relaxing study focus videos or late night coding sessions",
        'default_flowstep': 0.12,
        'steps': 25
    },
    'boom_bap': {
        'prompt': "Classic 90s boom bap hip hop drums with punchy kicks, crisp snares, and jazz sample chops, perfect for documentary footage of urban street scenes and skateboarding",
        'default_flowstep': 0.12,
        'steps': 25
    },
    'percussion_ensemble': {
        'prompt': "Rich percussive ensemble with djembe, congas, shakers, and tribal drums creating complex polyrhythms, perfect for nature documentaries about rainforests or ancient cultural rituals",
        'default_flowstep': 0.12,
        'steps': 25
    },

    # Enhanced Electronic Music
    'future_bass': {
        'prompt': "Energetic future bass with filtered supersaws, pitch-bending lead synths, heavy sidechain, and chopped vocal samples, perfect for extreme sports highlights or uplifting motivational content",
        'default_flowstep': 0.12,
        'steps': 25
    },
    'synthwave_retro': {
        'prompt': "80s retrofuturistic synthwave with gated reverb drums, analog arpeggios, neon-bright lead synths and driving bass, perfect for cyberpunk-themed technology showcases or retro gaming montages",
        'default_flowstep': 0.12,
        'steps': 25
    },
    'melodic_techno': {
        'prompt': "Hypnotic melodic techno with pulsing bass, atmospheric pads, and evolving synthesizer sequences with subtle filter modulation, ideal for timelapse footage of urban nightscapes or architectural showcases",
        'default_flowstep': 0.12,
        'steps': 25
    },
    'dubstep_wobble': {
        'prompt': "Heavy dubstep with aggressive wobble bass, metallic synthesizers, distorted drops, and tension-building risers, perfect for action sequence transitions or gaming highlight reels",
        'default_flowstep': 0.12,
        'steps': 25
    },

    # Glitchy Effects
    'glitch_hop': {
        'prompt': "Glitch hop with stuttering sample slices, bit-crushed percussion, granular synthesis textures and digital artifacts, perfect for technology malfunction scenes or data visualization animations",
        'default_flowstep': 0.12,
        'steps': 25
    },
    'digital_disruption': {
        'prompt': "Heavily glitched soundscape with digital artifacts, buffer errors, granular time stretching, and corrupted audio samples, ideal for cybersecurity themes or digital distortion transitions in tech presentations",
        'default_flowstep': 0.12,
        'steps': 25
    },
    'circuit_bent': {
        'prompt': "Circuit-bent toy sounds with unpredictable pitch shifts, broken electronic tones, and hardware malfunction artifacts, perfect for creative coding demonstrations or innovative technology exhibitions",
        'default_flowstep': 0.12,
        'steps': 25
    },

    # Experimental Hybrids
    'orchestral_glitch': {
        'prompt': "Cinematic orchestral elements disrupted by digital glitches, granular textures, and temporal distortions, perfect for science fiction trailers or futuristic product reveals with contrasting classical and modern elements",
        'default_flowstep': 0.12,
        'steps': 25
    },
    'vapor_drums': {
        'prompt': "Vaporwave drum processing with extreme pitch and time manipulation, reverb-drenched samples, and retro commercial music elements, ideal for nostalgic internet culture documentaries or retrofuturistic art installations",
        'default_flowstep': 0.12,
        'steps': 25
    },
    'industrial_textures': {
        'prompt': "Harsh industrial soundscape with mechanical percussion, factory recordings, metallic impacts, and distorted synth drones, perfect for manufacturing process videos or dystopian urban environments",
        'default_flowstep': 0.12,
        'steps': 25
    },
    'jungle_breaks': {
        'prompt': "High-energy jungle drum breaks with choppy breakbeat samples, deep sub bass, and dub reggae influences, perfect for fast-paced urban chase scenes or extreme sports montages",
        'default_flowstep': 0.12,
        'steps': 25
    }
}
