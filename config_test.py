import sys
from pathlib import Path

print('=== CONFIGURATION TEST ===')
project_root = Path('D:/codigos/rag/rag-1')
print('Project root:', project_root)

config_dir = project_root / 'config'
settings_file = config_dir / 'settings.py'

print('Config directory exists:', config_dir.exists())
print('Settings file exists:', settings_file.exists())

if settings_file.exists():
    sys.path.append(str(project_root))
    try:
        from config.settings import OLLAMA_CONFIG, DATA_PATHS, PROCESSING_CONFIG
        print('✅ SUCCESS: All imports work!')
        
        print('\nMODEL CONFIGURATION:')
        print('   Coder:', OLLAMA_CONFIG['models']['coder'])
        print('   Assistant:', OLLAMA_CONFIG['models']['assistant']) 
        print('   Embeddings:', OLLAMA_CONFIG['models']['embeddings'])
        
        print('\nDATA PATHS:')
        for name, path in DATA_PATHS.items():
            status = 'EXISTS' if path.exists() else 'MISSING'
            print(f'   {name}: {status}')
            
    except Exception as e:
        print('ERROR:', e)
        import traceback
        traceback.print_exc()
else:
    print('config/settings.py not found!')
