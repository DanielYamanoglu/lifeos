# LifeOS demo output

Embedder: hashing (offline). LLM: stub (offline). Stores: in-memory. All notes are fictional.

## 1. Ingest

Ingested 13 notes. Graph: {'Topic': 17, 'Person': 6, 'Project': 1, 'Note': 13, 'Event': 4, 'Task': 7, 'edges': 96}. Vector chunks: 13.

Re-ingesting the same files: 0 created, 12 skipped as duplicates. Vector chunks still 13.

## 2. Hybrid retrieval with provenance

### Query: What is Jonas working on?

1. **Pasta night plan** (score 0.0262)  
   Pasta Night is on Friday with Jonas and Priya. Jonas will bring the wine and Priya wants to try fresh tagliatelle. The sauce is a slow tomato ragu,...
   - vector: best chunk #0 cosine=0.19 (vector rank 1)
   - graph: query mentions 'Jonas'; 'Jonas' (mentioned in this note, 0 hops)
   - graph: top vector hit 'Pasta night plan' mentions 'San Marzano'; 'San Marzano' (mentioned in this note, 0 hops)
   - fusion: vector rank 1, graph rank 1 -> weighted RRF 0.0262
2. **Sourdough starter notes** (score 0.0255)  
   Feeding schedule for the starter: equal parts flour and water by weight, every 12 hours at room temperature. Maya gave me the starter and said a ry...
   - vector: best chunk #0 cosine=0.13 (vector rank 2)
   - graph: top vector hit 'Sourdough starter notes' mentions 'baking'; 'baking' (mentioned in this note, 0 hops)
   - graph: top vector hit 'Pasta night plan' mentions 'Alex'; 'Alex' <-ASSIGNED_TO- 'Buy rye flour' (mentioned in this note, 1 hop)
   - fusion: vector rank 2, graph rank 4 -> weighted RRF 0.0255
3. **Garden Bot sync notes** (score 0.0254)  
   Garden Bot Sync with Jonas on Wednesday. We decided to ship version one by the end of November: soil sensor, pump relay and the daily watering sche...
   - vector: best chunk #0 cosine=0.10 (vector rank 3)
   - graph: query mentions 'Jonas'; 'Jonas' (mentioned in this note, 0 hops)
   - graph: query mentions 'Jonas'; 'Jonas' -ATTENDS-> 'Garden Bot Sync' (mentioned in this note, 1 hop)
   - fusion: vector rank 3, graph rank 3 -> weighted RRF 0.0254
4. **Garden Bot design** (score 0.0253)  
   Garden Bot project: a small Raspberry Pi robot that waters the balcony plants when soil moisture drops below a threshold. The bot reports readings ...
   - vector: best chunk #0 cosine=0.05 (vector rank 4)
   - graph: query mentions 'Jonas'; 'Jonas' (mentioned in this note, 0 hops)
   - graph: query mentions 'Jonas'; 'Jonas' <-ASSIGNED_TO- 'Write sensor calibration script for the Garden Bot project' (mentioned in this note, 1 hop)
   - fusion: vector rank 4, graph rank 2 -> weighted RRF 0.0253

### Query: Who is helping with the trip?

1. **Lisbon trip plan** (score 0.0262)  
   Lisbon Trip is planned for early spring with Maya. We want four days, a day trip to Sintra and plenty of pastel de nata. #travel - [ ] Book flights...
   - vector: best chunk #0 cosine=0.38 (vector rank 1)
   - graph: top vector hit 'Lisbon trip plan' mentions 'Sintra'; 'Sintra' (mentioned in this note, 0 hops)
   - graph: top vector hit 'Lisbon trip plan' mentions 'Alfama'; 'Alfama' (mentioned in this note, 0 hops)
   - fusion: vector rank 1, graph rank 1 -> weighted RRF 0.0262
2. **Restaurant ideas** (score 0.0256)  
   Tomas recommended three small restaurants for the Lisbon Trip: a seafood place in Alfama, a tasca that has cheap daily specials and a bakery that o...
   - vector: best chunk #0 cosine=0.10 (vector rank 3)
   - graph: top vector hit 'Lisbon trip plan' mentions 'Alfama'; 'Alfama' (mentioned in this note, 0 hops)
   - graph: top vector hit 'Remember to ask Tomas about the Lisbon Trip tram...' mentions 'Tomas'; 'Tomas' (mentioned in this note, 0 hops)
   - fusion: vector rank 3, graph rank 2 -> weighted RRF 0.0256
3. **Remember to ask Tomas about the Lisbon Trip tram...** (score 0.0252)  
   Remember to ask Tomas about the Lisbon Trip tram pass. #travel
   - vector: best chunk #0 cosine=0.24 (vector rank 2)
   - graph: top vector hit 'Remember to ask Tomas about the Lisbon Trip tram...' mentions 'Tomas'; 'Tomas' (mentioned in this note, 0 hops)
   - graph: top vector hit 'Lisbon trip plan' mentions 'travel'; 'travel' (mentioned in this note, 0 hops)
   - fusion: vector rank 2, graph rank 6 -> weighted RRF 0.0252
4. **One-on-one with Maya** (score 0.0251)  
   Maya asked Alex to share the sourdough hydration ratio before the Lisbon Trip. We also agreed to split the apartment cost evenly and to set a daily...
   - vector: best chunk #0 cosine=0.09 (vector rank 4)
   - graph: top vector hit 'Lisbon trip plan' mentions 'Maya'; 'Maya' (mentioned in this note, 0 hops)
   - graph: top vector hit 'Lisbon trip plan' mentions 'travel'; 'travel' (mentioned in this note, 0 hops)
   - fusion: vector rank 4, graph rank 3 -> weighted RRF 0.0251

### Query: What did Lena recommend?

1. **Book club notes** (score 0.0261)  
   Book Club Meeting with Lena on Thursday evening. The group picked *Braiding Sweetgrass* for next month and moved the next date to the second Thursd...
   - vector: best chunk #0 cosine=0.10 (vector rank 1)
   - graph: query mentions 'Lena'; 'Lena' (mentioned in this note, 0 hops)
   - graph: query mentions 'Lena'; 'Lena' -ATTENDS-> 'Book Club Meeting' (mentioned in this note, 1 hop)
   - fusion: vector rank 1, graph rank 2 -> weighted RRF 0.0261
2. **Reading list** (score 0.0260)  
   Lena recommended *The Overstory* last month and it is first on the list. Next are *Braiding Sweetgrass* and *The Soul of an Octopus*. Aim for two b...
   - vector: best chunk #0 cosine=0.09 (vector rank 2)
   - graph: query mentions 'Lena'; 'Lena' (mentioned in this note, 0 hops)
   - graph: query mentions 'Lena'; 'Lena' <-RELATED_TO- 'The Overstory' (mentioned in this note, 1 hop)
   - fusion: vector rank 2, graph rank 1 -> weighted RRF 0.0260
3. **Balcony herbs** (score 0.0159)  
   Tomas gave me basil and mint seeds for the balcony. In summer the basil should be watered daily, so Garden Bot will handle the watering once versio...
   - vector: best chunk #0 cosine=0.00 (vector rank 3)
   - fusion: vector rank 3 -> weighted RRF 0.0159
4. **Garden Bot design** (score 0.0156)  
   Garden Bot project: a small Raspberry Pi robot that waters the balcony plants when soil moisture drops below a threshold. The bot reports readings ...
   - vector: best chunk #0 cosine=0.00 (vector rank 4)
   - fusion: vector rank 4 -> weighted RRF 0.0156

## 3. Graph lookup: `get_entity('Garden Bot')`

Garden Bot (Project); mentioned in 5 notes.

- <- PART_OF Write sensor calibration script for the Garden Bot project [Task] assigned_to=['Jonas']
- <- PART_OF Order a 5V relay board for the Garden Bot project [Task] assigned_to=['Alex']
- -> RELATED_TO Raspberry Pi [Topic]
- <- WORKS_ON Jonas [Person]
- <- WORKS_ON Alex [Person]

## 4. Agents (stub LLM, tool-calling loop)

### capture agent

Input: Jonas suggested a waterproof case for Garden Bot. Alex will order it. #electronics

Tool calls: `add_note` (2 model turns)

> Note 'Jonas suggested a waterproof case for Garden Bot....' stored. Categorised entities -> Person: Jonas, Alex; Topic: electronics, Garden Bot.

### answer agent

Input: What is Jonas working on?

Tool calls: `search_memory` (2 model turns)

> From memory (extractive stub answer):
> - Pasta night plan: Pasta Night is on Friday with Jonas and Priya. [why: vector: best chunk #0 cosine=0.19 (vector rank 2)]
> - Jonas suggested a waterproof case for Garden Bot....: Jonas suggested a waterproof case for Garden Bot. [why: vector: best chunk #0 cosine=0.29 (vector rank 1)]
> - Garden Bot sync notes: Garden Bot Sync with Jonas on Wednesday. [why: vector: best chunk #0 cosine=0.10 (vector rank 4)]

### planner agent

Input: Plan the Lisbon Trip

Tool calls: `search_memory`, `get_entity` (3 model turns)

> Plan for 'Lisbon Trip' from open tasks in memory:
> 1. Book flights for the Lisbon Trip (owner: Maya)
> 2. Reserve an apartment near Alfama for the Lisbon Trip (owner: Alex)

