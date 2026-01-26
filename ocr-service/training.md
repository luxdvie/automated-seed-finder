# OCR Template Training

Extract each template when that element is visible on the map screen.

## Nightlord Templates

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

## Shifting Earth Templates

Extract when that shifting earth event is visible on the map:

1. [MountainTop](http://localhost:8000/extract-shifting-earth/2?name=MountainTop) - White area, top-left
2. [Crater](http://localhost:8000/extract-shifting-earth/2?name=Crater) - Red/volcano, middle-top
3. [Noklateo](http://localhost:8000/extract-shifting-earth/2?name=Noklateo) - Blue/silver, bottom-left
4. [RottedWoods](http://localhost:8000/extract-shifting-earth/2?name=RottedWoods) - Pink/barren, bottom-right
5. [GreatHollow](http://localhost:8000/extract-shifting-earth/2?name=GreatHollow) - Entirely different map

## After Training

Restart the service to reload all templates:
```bash
cd ocr-service && ./venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

Test detection:
```
http://localhost:8000/capture-monitor/3
http://localhost:8000/debug-shifting-earth/3
```
