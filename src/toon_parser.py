import re
import logging
import csv
import json
from io import StringIO

logger = logging.getLogger(__name__)

def parse_toon_line(line_def, data_line):
    """
    Parses a single TOON data line based on headers.
    Handles CSV-style quoting for text fields.
    Robustly handles '9/10' or '(9)' formats in numeric fields.
    """
    if not data_line or data_line.isspace():
        return {}

    try:
        # Use csv module to handle quoted strings
        reader = csv.reader(StringIO(data_line), skipinitialspace=True)
        try:
            values = next(reader)
        except StopIteration:
            values = []
        
        cleaned_values = []
        for v in values:
            v_str = v.strip()
            # Remove parens: (9) -> 9
            v_str = v_str.replace('(', '').replace(')', '')
            # Handle fractional scores: 9/10 -> 9
            if '/' in v_str and any(c.isdigit() for c in v_str):
                parts = v_str.split('/')
                # If first part is digit, take it. 
                if parts[0].strip().isdigit():
                    v_str = parts[0].strip()
            cleaned_values.append(v_str)

        headers = line_def.get('headers', [])
        
        # Ensure values match headers length if possible, or pad
        if len(cleaned_values) < len(headers):
            cleaned_values += [""] * (len(headers) - len(cleaned_values))
        elif len(cleaned_values) > len(headers):
            cleaned_values = cleaned_values[:len(headers)]

        return dict(zip(headers, cleaned_values))
    except Exception as e:
        logger.error(f"Error parsing TOON line '{data_line}': {e}")
        return {}

def fuzzy_extract_scores(text: str) -> dict:
    """
    Fallback method. Scans text for key metrics followed near-immediately by a number.
    Handles: "Visual: 9", "Visual - 9", "Visual: 9/10", "Accuracy: 9/10"
    """
    scores = {
        'visual': '0', 'audio': '0', 'source': '0', 'logic': '0', 'emotion': '0',
        'video_audio': '0', 'video_caption': '0', 'audio_caption': '0'
    }
    
    # Mappings: Regex Pattern -> Score Key
    mappings = [
        ('visual', 'visual'),
        ('visual.*?integrity', 'visual'),
        ('accuracy', 'visual'), # Fallback
        ('audio', 'audio'),
        ('source', 'source'),
        ('logic', 'logic'),
        ('emotion', 'emotion'),
        (r'video.*?audio', 'video_audio'),
        (r'video.*?caption', 'video_caption'),
        (r'audio.*?caption', 'audio_caption')
    ]

    for pattern_str, key in mappings:
        pattern = re.compile(fr'(?i){pattern_str}.*?[:=\-\s\(]+(\b10\b|\b\d\b)(?:/10)?')
        match = pattern.search(text)
        if match:
            if scores[key] == '0':
                scores[key] = match.group(1)
    
    return scores

def parse_veracity_toon(text: str) -> dict:
    """
    Parses the Veracity Vector TOON output into a standardized dictionary.
    Handles "Simple", "Reasoning", and new "Modalities" blocks.
    Robust against Markdown formatting artifacts and nested reports.
    Also handles JSON fallback if the model outputs JSON instead of TOON.
    """
    if not text:
        return {}

    # Initialize Flat Result
    flat_result = {
        'veracity_vectors': {
            'visual_integrity_score': '0',
            'audio_integrity_score': '0',
            'source_credibility_score': '0',
            'logical_consistency_score': '0',
            'emotional_manipulation_score': '0'
        },
        'modalities': {
            'video_audio_score': '0',
            'video_caption_score': '0',
            'audio_caption_score': '0'
        },
        'video_context_summary': '',
        'tags': [],
        'factuality_factors': {
            'claim_accuracy': 'Unverifiable',
            'evidence_gap': '',
            'grounding_check': ''
        },
        'disinformation_analysis': {
            'classification': 'None',
            'intent': 'None',
            'threat_vector': 'None'
        },
        'final_assessment': {
            'veracity_score_total': '0',
            'reasoning': ''
        }
    }

    # Clean text
    clean_text = re.sub(r'```\w*', '', text)
    clean_text = re.sub(r'```', '', clean_text)
    clean_text = clean_text.strip()

    parsed_sections = {}

    # --- STRATEGY 1: Strict Block Regex (Original) ---
    # Matches: key : type [ count ] { headers } :
    block_pattern = re.compile(
        r'([a-zA-Z0-9_]+)\s*:\s*(?:\w+\s*)?(?:\[\s*(\d+)\s*\])?\s*\{\s*(.*?)\s*\}\s*:\s*', 
        re.MULTILINE
    )
    
    matches = list(block_pattern.finditer(clean_text))
    toon_success = False
    
    if matches:
        toon_success = True
        for i, match in enumerate(matches):
            key = match.group(1).lower()
            count = int(match.group(2)) if match.group(2) else 1
            headers_str = match.group(3)
            headers = [h.strip().lower() for h in headers_str.split(',')]
            
            start_idx = match.end()
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(clean_text)
            block_content = clean_text[start_idx:end_idx].strip()
            
            lines = [line.strip() for line in block_content.splitlines() if line.strip()]
            valid_lines = [l for l in lines if len(l) > 1] 
            
            data_items = []
            for line in valid_lines[:max(1, count)]:
                item = parse_toon_line({'key': key, 'headers': headers}, line)
                data_items.append(item)
                
            if count == 1 and data_items:
                parsed_sections[key] = data_items[0]
            else:
                parsed_sections[key] = data_items

    # --- STRATEGY 2: Flexible Line Scanner (New Robust Fallback) ---
    # Used if strict regex finds nothing, but text looks like TOON (line-based)
    if not toon_success:
        lines = clean_text.splitlines()
        current_key = None
        KNOWN_KEYS = {'summary', 'tags', 'vectors', 'modalities', 'factuality', 'disinfo', 'final'}
        
        temp_sections = {}
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Check for header "key:"
            potential_key = line.split(':')[0].strip().lower()
            if potential_key in KNOWN_KEYS:
                current_key = potential_key
                if current_key not in temp_sections: temp_sections[current_key] = []
                
                # Check for inline value "key: value"
                if ':' in line:
                    val = line.split(':', 1)[1].strip()
                    if val:
                        # Special handling for tags array syntax
                        if val.startswith('[') and val.endswith(']'):
                            try:
                                val_list = json.loads(val)
                                temp_sections[current_key] = val_list
                            except:
                                temp_sections[current_key] = [val]
                        # Special handling for final: assessment(...)
                        elif current_key == 'final' and val.startswith('assessment'):
                            val_content = val.replace('assessment', '', 1).strip()
                            if val_content.startswith('(') and val_content.endswith(')'):
                                val_content = val_content[1:-1]
                            temp_sections[current_key] = parse_toon_line({'headers': ['score', 'reasoning']}, val_content)
                        else:
                            # It's a string value, treat as single item dict or direct text
                            if current_key == 'summary':
                                temp_sections[current_key] = {'text': val}
                            else:
                                temp_sections[current_key].append(parse_toon_line({}, val))
                continue
            
            if current_key:
                # Handle YAML style "Subkey: Value" lines for Vectors/Modalities
                # Example: "Visual: (7/10, ...)"
                if ':' in line and current_key in ['vectors', 'modalities']:
                    parts = line.split(':', 1)
                    subkey = parts[0].strip() # e.g. "Visual"
                    subval = parts[1].strip() # e.g. "(7/10, ...)"
                    
                    # Convert to comma-separated line for parse_toon_line
                    # Remove parens if present
                    if subval.startswith('(') and subval.endswith(')'):
                        subval = subval[1:-1]
                    
                    # Construct pseudo-CSV: "Visual, 7/10, ..."
                    pseudo_line = f"{subkey}, {subval}"
                    headers = ['category', 'score', 'reasoning']
                    
                    # Ensure container is list
                    if isinstance(temp_sections[current_key], dict): 
                        temp_sections[current_key] = [temp_sections[current_key]]
                    
                    temp_sections[current_key].append(parse_toon_line({'headers': headers}, pseudo_line))
                    continue

                if ':' not in line and not line.startswith('{'):
                    # Data line for current key (Standard TOON)
                    headers = []
                    if current_key in ['vectors', 'modalities']:
                        headers = ['category', 'score', 'reasoning']
                    elif current_key == 'disinfo':
                        headers = ['class', 'intent', 'threat']
                    elif current_key == 'final':
                        headers = ['score', 'reasoning']
                    elif current_key == 'factuality':
                        headers = ['accuracy', 'gap', 'grounding']
                    
                    if isinstance(temp_sections[current_key], dict):
                        temp_sections[current_key] = [temp_sections[current_key]]
                    
                    if isinstance(temp_sections[current_key], list):
                        temp_sections[current_key].append(parse_toon_line({'headers': headers}, line))
                    
        if temp_sections:
            parsed_sections = temp_sections
            toon_success = True

    # --- STRATEGY 3: JSON Parsing Fallback ---
    if not toon_success or ('vectors' not in parsed_sections and 'final' not in parsed_sections):
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                json_data = json.loads(json_str)
                parsed_sections = json_data
                toon_success = True
        except Exception as e:
            logger.debug(f"JSON Parsing failed: {e}")

    # --- Flatten logic ---
    
    got_vectors = False
    got_modalities = False

    # 1. Process 'vectors'
    vectors_data = parsed_sections.get('vectors', [])
    if isinstance(vectors_data, dict): 
        v = vectors_data
        if 'visual' in v: flat_result['veracity_vectors']['visual_integrity_score'] = str(v['visual'])
        if 'audio' in v: flat_result['veracity_vectors']['audio_integrity_score'] = str(v['audio'])
        if 'source' in v: flat_result['veracity_vectors']['source_credibility_score'] = str(v['source'])
        if 'logic' in v: flat_result['veracity_vectors']['logical_consistency_score'] = str(v['logic'])
        if 'emotion' in v: flat_result['veracity_vectors']['emotional_manipulation_score'] = str(v['emotion'])
        if any(str(val) != '0' for val in v.values()):
            got_vectors = True

    elif isinstance(vectors_data, list): 
        for item in vectors_data:
            if not isinstance(item, dict): continue
            cat = item.get('category', '').lower()
            score = str(item.get('score', '0'))
            if score and score != '0': 
                got_vectors = True
            if 'visual' in cat: flat_result['veracity_vectors']['visual_integrity_score'] = score
            elif 'audio' in cat: flat_result['veracity_vectors']['audio_integrity_score'] = score
            elif 'source' in cat: flat_result['veracity_vectors']['source_credibility_score'] = score
            elif 'logic' in cat: flat_result['veracity_vectors']['logical_consistency_score'] = score
            elif 'emotion' in cat: flat_result['veracity_vectors']['emotional_manipulation_score'] = score

    # 2. Process 'modalities'
    modalities_data = parsed_sections.get('modalities', [])
    if isinstance(modalities_data, dict):
        m = modalities_data
        for k, v in m.items():
            k_clean = k.lower().replace(' ', '').replace('-', '').replace('_', '')
            if 'videoaudio' in k_clean: flat_result['modalities']['video_audio_score'] = str(v)
            elif 'videocaption' in k_clean: flat_result['modalities']['video_caption_score'] = str(v)
            elif 'audiocaption' in k_clean: flat_result['modalities']['audio_caption_score'] = str(v)
            if str(v) != '0': got_modalities = True

    elif isinstance(modalities_data, list):
        for item in modalities_data:
            if not isinstance(item, dict): continue
            cat = item.get('category', '').lower().replace(' ', '').replace('-', '').replace('_', '')
            score = str(item.get('score', '0'))
            if score and score != '0':
                got_modalities = True
            if 'videoaudio' in cat: flat_result['modalities']['video_audio_score'] = score
            elif 'videocaption' in cat: flat_result['modalities']['video_caption_score'] = score
            elif 'audiocaption' in cat: flat_result['modalities']['audio_caption_score'] = score

    # --- FUZZY FALLBACK ---
    if not got_vectors or not got_modalities:
        fuzzy_scores = fuzzy_extract_scores(text)
        if not got_vectors:
            flat_result['veracity_vectors']['visual_integrity_score'] = fuzzy_scores['visual']
            flat_result['veracity_vectors']['audio_integrity_score'] = fuzzy_scores['audio']
            flat_result['veracity_vectors']['source_credibility_score'] = fuzzy_scores['source']
            flat_result['veracity_vectors']['logical_consistency_score'] = fuzzy_scores['logic']
            flat_result['veracity_vectors']['emotional_manipulation_score'] = fuzzy_scores['emotion']
        if not got_modalities:
            flat_result['modalities']['video_audio_score'] = fuzzy_scores['video_audio']
            flat_result['modalities']['video_caption_score'] = fuzzy_scores['video_caption']
            flat_result['modalities']['audio_caption_score'] = fuzzy_scores['audio_caption']

    # 3. Factuality
    f = parsed_sections.get('factuality', {})
    if isinstance(f, list): f = f[0] if f else {}
    if f:
        flat_result['factuality_factors']['claim_accuracy'] = f.get('accuracy', 'Unverifiable')
        flat_result['factuality_factors']['evidence_gap'] = f.get('gap', '')
        flat_result['factuality_factors']['grounding_check'] = f.get('grounding', '')

    # 4. Disinfo
    d = parsed_sections.get('disinfo', {})
    if isinstance(d, list): d = d[0] if d else {}
    if d:
        flat_result['disinformation_analysis']['classification'] = d.get('class', 'None')
        flat_result['disinformation_analysis']['intent'] = d.get('intent', 'None')
        flat_result['disinformation_analysis']['threat_vector'] = d.get('threat', 'None')

    # 5. Final Assessment
    fn = parsed_sections.get('final', {})
    if isinstance(fn, list): fn = fn[0] if fn else {}
    if fn:
        flat_result['final_assessment']['veracity_score_total'] = str(fn.get('score', '0'))
        flat_result['final_assessment']['reasoning'] = fn.get('reasoning', '')

    # 6. Tags
    t = parsed_sections.get('tags', [])
    if isinstance(t, list) and t and isinstance(t[0], str):
        flat_result['tags'] = t
    elif isinstance(t, list) and t and isinstance(t[0], dict):
        raw_tags = t[0].get('keywords', '')
        if raw_tags:
            flat_result['tags'] = [x.strip() for x in raw_tags.split(',')]
    elif isinstance(t, dict):
        raw_tags = t.get('keywords', '')
        if raw_tags:
             flat_result['tags'] = [x.strip() for x in raw_tags.split(',')]

    # 7. Summary
    s = parsed_sections.get('summary', {})
    if isinstance(s, list): s = s[0] if s else {}
    if isinstance(s, dict):
        if 'text' in s: flat_result['video_context_summary'] = s['text']
        else:
             for k,v in s.items():
                 if isinstance(v, str): flat_result['video_context_summary'] = v; break
    elif isinstance(s, str):
        flat_result['video_context_summary'] = s

    flat_result['raw_parsed_structure'] = parsed_sections
    
    return flat_result
