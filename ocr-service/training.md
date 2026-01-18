# Nightlord Template Training

Extract each template when that nightlord is visible on the map screen.

## Templates to Extract

1. ~~Tricephalos~~ (done)
2. [Augur](http://localhost:8000/extract-nightlord/3?name=Augur)
3. [GapingJaw](http://localhost:8000/extract-nightlord/3?name=GapingJaw)
4. [SentientPest](http://localhost:8000/extract-nightlord/3?name=SentientPest)
5. [DarkdriftKnight](http://localhost:8000/extract-nightlord/3?name=DarkdriftKnight)
6. [EquilibrousBeast](http://localhost:8000/extract-nightlord/3?name=EquilibrousBeast)
7. [Balancers](http://localhost:8000/extract-nightlord/3?name=Balancers)
8. [Dreglord](http://localhost:8000/extract-nightlord/3?name=Dreglord)
9. [FissureInTheFog](http://localhost:8000/extract-nightlord/3?name=FissureInTheFog)
10. [NightAspect](http://localhost:8000/extract-nightlord/3?name=NightAspect)

## After Training

Restart the service to reload all templates:
```bash
cd ocr-service && ./venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

Test detection:
```
http://localhost:8000/capture-monitor/3
```
