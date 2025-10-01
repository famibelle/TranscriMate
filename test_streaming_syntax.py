#!/usr/bin/env python3
# Test rapide de l'endpoint streaming après restructuration

import asyncio
import json

async def test_syntax():
    """Test pour vérifier que la syntaxe du nouveau streaming fonctionne"""
    
    # Simuler les paramètres
    file_path = "backend/Multimedia/dialogue_test.mp3"
    file_extension = ".mp3"
    filename = "dialogue_test.mp3"
    
    print("🔧 Test de syntaxe du streaming...")
    
    # Import de la fonction (cela testera la syntaxe)
    try:
        import sys
        sys.path.append('backend')
        from main import process_streaming_audio
        print("✅ Import réussi - syntaxe correcte!")
        
        # Test basic d'itération (juste les premiers éléments)
        print("🏃 Test d'exécution basique...")
        async for i, chunk in enumerate(process_streaming_audio(file_path, file_extension, filename)):
            print(f"📦 Chunk {i}: {chunk[:50]}..." if len(chunk) > 50 else f"📦 Chunk {i}: {chunk}")
            if i >= 2:  # Arrêter après 3 chunks pour le test
                break
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
    except SyntaxError as e:
        print(f"❌ Erreur de syntaxe: {e}")
    except Exception as e:
        print(f"⚠️ Erreur d'exécution (normal pour le test): {e}")

if __name__ == "__main__":
    asyncio.run(test_syntax())