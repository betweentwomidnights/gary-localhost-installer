"""
Riff Manager - Handles BPM stretching and key selection for style transfer
Uses librosa (already installed) to avoid new dependencies
"""

import librosa
import soundfile as sf
import numpy as np
import os
import re
from typing import Dict, List, Optional, Tuple
import tempfile
import random

class RiffManager:
    def __init__(self, riffs_directory: str = "/app/riffs"):
        self.riffs_directory = riffs_directory
        self.riff_library = {}
        self.target_duration = 12.0  # seconds for style transfer
        self._scan_riffs()
    
    def _scan_riffs(self):
        """Scan the riffs directory and build the library"""
        if not os.path.exists(self.riffs_directory):
            print(f"⚠️  Riffs directory not found: {self.riffs_directory}")
            return
        
        print(f"🔍 Scanning riffs in {self.riffs_directory}")
        
        for filename in os.listdir(self.riffs_directory):
            if filename.endswith('.wav'):
                riff_info = self._parse_filename(filename)
                if riff_info:
                    key = riff_info['key']
                    if key not in self.riff_library:
                        self.riff_library[key] = []
                    
                    riff_info['file_path'] = os.path.join(self.riffs_directory, filename)
                    self.riff_library[key].append(riff_info)
        
        # Sort riffs by BPM within each key
        for key in self.riff_library:
            self.riff_library[key].sort(key=lambda x: x['original_bpm'])
        
        print(f"✅ Found riffs for {len(self.riff_library)} keys:")
        for key, riffs in self.riff_library.items():
            print(f"   {key}: {len(riffs)} riffs ({[r['original_bpm'] for r in riffs]} BPM)")
    
    def _parse_filename(self, filename: str) -> Optional[Dict]:
        """
        Parse filenames like: gsharp_80bpm_4.wav, fsharp_83_bpm_1.wav, f_91bpm_5.wav
        """
        # Remove .wav extension
        name = filename.replace('.wav', '')
        
        # Pattern: key_bpm[_number]
        # Handle both "80bpm" and "83_bpm" formats
        pattern = r'^([a-g](?:sharp)?)_(\d+)(?:_)?bpm(?:_\d+)?$'
        match = re.match(pattern, name, re.IGNORECASE)
        
        if match:
            key = match.group(1).lower()
            bpm = int(match.group(2))
            
            return {
                'filename': filename,
                'key': key,
                'original_bpm': bpm,
                'description': f'{key} riff at {bpm} BPM'
            }
        else:
            print(f"⚠️  Could not parse filename: {filename}")
            return None
    
    def get_available_keys(self) -> List[str]:
        """Get all available keys"""
        return sorted(self.riff_library.keys())
    
    def get_riffs_for_key(self, key: str) -> List[Dict]:
        """Get all riffs for a specific key"""
        return self.riff_library.get(key.lower(), [])
    
    def select_best_riff(self, key: str, target_bpm: int, random_selection: bool = True) -> Optional[Dict]:
        """
        Select a riff for a given key - now just random selection from all available!
        
        Args:
            key: Musical key
            target_bpm: Target BPM (not used for selection anymore, just for logging)
            random_selection: If True, randomly pick from all riffs. If False, pick first one.
        
        Returns:
            Selected riff info or None
        """
        riffs = self.get_riffs_for_key(key)
        if not riffs:
            return None
        
        if len(riffs) == 1:
            print(f"🎯 Only one option: {riffs[0]['filename']}")
            return riffs[0]
        
        if random_selection:
            # Just randomly pick from ALL riffs for this key!
            selected_riff = random.choice(riffs)
            
            stretch_ratio = target_bpm / selected_riff['original_bpm']
            print(f"🎲 Random selection: {selected_riff['filename']} from {len(riffs)} total options")
            print(f"   Will stretch {selected_riff['original_bpm']} → {target_bpm} BPM ({stretch_ratio:.2f}x)")
            print(f"   All options: {[r['filename'] for r in riffs]}")
            
            return selected_riff
        
        else:
            # Non-random: just pick the first one (for testing/debugging)
            selected_riff = riffs[0]
            stretch_ratio = target_bpm / selected_riff['original_bpm']
            print(f"🎯 First option: {selected_riff['filename']} ({stretch_ratio:.2f}x stretch)")
            return selected_riff
    
    def stretch_riff_to_bpm(self, riff_info: Dict, target_bpm: int) -> str:
        """
        Stretch a riff to target BPM and return temp file path
        """
        file_path = riff_info['file_path']
        original_bpm = riff_info['original_bpm']
        
        print(f"🎸 Processing riff: {riff_info['filename']}")
        print(f"   {original_bpm} BPM → {target_bpm} BPM (ratio: {target_bpm/original_bpm:.3f}x)")
        
        # Load audio
        audio, sr = librosa.load(file_path, sr=None)
        original_duration = len(audio) / sr
        
        # Calculate stretch rate
        stretch_rate = target_bpm / original_bpm
        
        # Time stretch
        stretched_audio = librosa.effects.time_stretch(audio, rate=stretch_rate)
        stretched_duration = len(stretched_audio) / sr
        
        # Slice to 12 seconds for style transfer
        target_samples = int(self.target_duration * sr)
        if len(stretched_audio) > target_samples:
            sliced_audio = stretched_audio[:target_samples]
            print(f"✂️  Sliced to {self.target_duration}s for style transfer")
        else:
            sliced_audio = stretched_audio
            actual_duration = len(sliced_audio) / sr
            print(f"⚠️  Audio only {actual_duration:.2f}s (target: {self.target_duration}s)")
        
        # Save to temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='stretched_riff_')
        os.close(temp_fd)  # Close the file descriptor, keep the path
        
        sf.write(temp_path, sliced_audio, sr)
        
        print(f"✅ Stretched riff saved to temp file")
        return temp_path
    
    def get_riff_for_style_transfer(self, key: str, target_bpm: int, random_selection: bool = True) -> Optional[str]:
        """
        Get a riff stretched to the target BPM, ready for style transfer
        Returns path to temporary file (caller should clean up)
        
        Args:
            key: Musical key
            target_bpm: Target BPM
            random_selection: If True, randomly select from equally good riffs
        """
        # Select best riff for this key/BPM combo (now with randomness!)
        riff_info = self.select_best_riff(key, target_bpm, random_selection)
        if not riff_info:
            print(f"❌ No riffs found for key: {key}")
            return None
        
        # Stretch to target BPM
        try:
            temp_path = self.stretch_riff_to_bpm(riff_info, target_bpm)
            return temp_path
        except Exception as e:
            print(f"❌ Failed to stretch riff: {e}")
            return None
    
    def list_library(self):
        """Print the entire riff library for debugging"""
        print("🎵 Riff Library:")
        for key, riffs in self.riff_library.items():
            print(f"\n{key.upper()}:")
            for riff in riffs:
                print(f"  - {riff['filename']} ({riff['original_bpm']} BPM)")