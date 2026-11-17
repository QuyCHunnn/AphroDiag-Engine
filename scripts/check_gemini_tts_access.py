"""Read an API key without echo; print only TTS model IDs or sanitized status."""
import getpass,json,urllib.request,urllib.error
key=getpass.getpass('Gemini API key (not saved): ')
try:
 request=urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000',headers={'x-goog-api-key':key})
 with urllib.request.urlopen(request,timeout=30) as response:payload=json.load(response)
 models=[m['name'] for m in payload.get('models',[]) if 'tts' in m.get('name','').lower()]
 print(json.dumps({'status':'success','tts_models':models},indent=2))
except urllib.error.HTTPError as exc:
 code=exc.code;exc.close();print(json.dumps({'status':'api_rejected','http_status':code}))
except Exception:
 print(json.dumps({'status':'connection_failed'}))
finally:key=None
