#!/usr/bin/env python3
import requests
import json
import time

def test_transcribe_streaming():
    """Test de l'endpoint Mode 2 - Streaming avec Server-Sent Events"""
    url = "http://localhost:8000/transcribe_streaming/"
    file_path = r"backend\Multimedia\dialogue_test.mp3"
    
    print(f"🎵 Test du Mode 2 - Streaming SSE")
    print(f"📁 Fichier: {file_path}")
    print(f"🔗 URL: {url}")
    print("⏳ Envoi du fichier avec streaming en cours...")
    print("=" * 60)
    
    try:
        with open(file_path, 'rb') as audio_file:
            files = {'file': (file_path, audio_file, 'audio/mpeg')}
            
            # Utiliser stream=True pour recevoir les Server-Sent Events
            response = requests.post(url, files=files, stream=True, timeout=120)
            
        print(f"📊 Statut HTTP: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Connexion streaming établie!")
            print("📡 Reception des événements en temps réel:\n")
            
            chunk_count = 0
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    chunk_count += 1
                    print(f"📦 Chunk #{chunk_count}:")
                    
                    # Essayer de parser le JSON
                    try:
                        data = json.loads(line)
                        
                        # Affichage formaté selon le type de message
                        if 'extraction_audio_status' in data:
                            print(f"   🎵 {data['message']}")
                        elif 'status' in data and data['status'] == 'diarization_processing':
                            print(f"   🎯 {data['message']}")
                        elif 'status' in data and data['status'] == 'diarization_done':
                            print(f"   ✅ {data['message']}")
                        elif 'diarization' in data:
                            speakers = len(set(seg['speaker'] for seg in data['diarization']))
                            segments = len(data['diarization'])
                            print(f"   📊 Diarisation: {speakers} locuteurs, {segments} segments")
                        elif 'speaker' in data and 'text' in data:
                            speaker = data['speaker']
                            if isinstance(data['text'], dict) and 'text' in data['text']:
                                text = data['text']['text'].strip()
                            else:
                                text = str(data['text']).strip()
                            start_time = data.get('start_time', 0)
                            print(f"   🗣️ {speaker} ({start_time:.1f}s): {text}")
                        else:
                            # Message générique
                            print(f"   📄 {json.dumps(data, ensure_ascii=False)}")
                            
                    except json.JSONDecodeError:
                        # Si ce n'est pas du JSON, afficher tel quel
                        print(f"   📝 {line}")
                    
                    print()  # Ligne vide pour la lisibilité
            
            print(f"🏁 Streaming terminé - {chunk_count} chunks reçus")
                        
        else:
            print("❌ Erreur!")
            print(f"Réponse: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transcribe_streaming()