# Companion Interaction Tracking System Tests

## Overview
The Companion Interaction Tracking System enables the AI companion to track, plan, and generate realistic outcomes for interactions with third parties based on their relationship dynamics.

## Test Scenarios

### 1. Planning Future Interactions
```bash
# Plan a coffee meeting with Alice
curl -X POST http://localhost:3000/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I plan to meet Alice for coffee tomorrow"}'
```
Expected: System detects the planned interaction and stores it for future reference

### 2. Inquiring About Past Interactions
```bash
# Ask about a past interaction
curl -X POST http://localhost:3000/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Did you meet with Alice yesterday?"}'
```
Expected: System generates realistic outcome based on relationship quality with Alice

### 3. Manual Interaction Planning
```bash
# Create a planned interaction via API
curl -X POST http://localhost:3000/api/interactions/plan \
  -H "Content-Type: application/json" \
  -d '{
    "third_party_id": 1,
    "companion_id": 1,
    "interaction_type": "planned",
    "description": "Have lunch with Bob",
    "planned_date": "tomorrow",
    "impact_on_relationship": 0.0
  }'
```
Expected: Returns interaction_id for the planned interaction

### 4. Completing Interactions
```bash
# Complete a planned interaction
curl -X POST http://localhost:3000/api/interactions/1/complete
```
Expected: Generates realistic outcome based on current attitudes and updates relationship

### 5. Viewing Interaction History
```bash
# Get interaction history between companion and third party
curl -X GET http://localhost:3000/api/interactions/history/1/2
```
Expected: Returns array of past interactions with outcomes

### 6. Viewing Planned Interactions
```bash
# Get all planned interactions for companion
curl -X GET http://localhost:3000/api/interactions/planned/1
```
Expected: Returns list of upcoming planned interactions

### 7. Detecting Interaction Requests
```bash
# Detect interaction in message
curl -X POST http://localhost:3000/api/interactions/detect \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Have you seen John recently?",
    "companion_id": 1
  }'
```
Expected: Detects query about John and returns relevant interaction if exists

## Outcome Generation Examples

### High Relationship Quality (>50)
- Coffee/Lunch: "Had a wonderful time with Alice! We talked about various topics and really enjoyed each other's company."
- Phone Call: "Had a great phone conversation with Bob. We caught up on recent events and shared some laughs."
- Help: "Sarah was incredibly grateful for my help! They thanked me multiple times."

### Medium Relationship Quality (0-50)
- Coffee/Lunch: "Met with Alice as planned. The conversation was pleasant enough, though there were a few awkward moments."
- Phone Call: "Spoke with Bob on the phone briefly. The conversation was polite but somewhat formal."
- Help: "Sarah appreciated the help, though they seemed a bit hesitant to accept it at first."

### Low Relationship Quality (<0)
- Coffee/Lunch: "The meeting with Alice was tense. We struggled to find common ground."
- Phone Call: "The phone call with Bob was brief and uncomfortable."
- Help: "Sarah reluctantly accepted my help but didn't seem very appreciative."

## Attitude Impact System

### Positive Interactions
- Fun/Enjoyable: +5 to +10 impact, increases joy and attraction
- Helping: +8 to +15 impact, increases gratitude and trust
- Meaningful: +3 to +8 impact, increases empathy and respect

### Negative Interactions
- Arguments: -10 to -15 impact, increases anger, decreases trust
- Disappointments: -5 to -10 impact, increases sorrow, decreases respect
- Betrayals: -15 to -20 impact, increases suspicion and disgust, destroys trust

## Integration Flow

1. **User mentions interaction** → System detects via pattern matching
2. **System checks for existing person** → Creates if new, updates if existing
3. **For past queries**: Generates outcome based on attitudes
4. **For future plans**: Stores interaction for later
5. **Updates attitudes** based on interaction outcome
6. **Provides context** to LLM for natural responses

## Success Criteria

✅ Planned interactions are stored and retrievable
✅ Past interaction queries generate realistic outcomes
✅ Outcomes vary based on relationship quality
✅ Attitudes update appropriately after interactions
✅ Integration with message processing is seamless
✅ API endpoints provide full CRUD operations
✅ Interaction history is maintained per person
✅ Context is provided to LLM for natural responses

## Example Conversation Flow

```
User: "I'm planning to have coffee with Alice tomorrow"
System: [Detects and stores planned interaction]
AI: "That sounds nice! I hope you have a good time with Alice."

[Next day]
User: "How did the coffee with Alice go?"
System: [Generates outcome based on relationship with Alice]
AI: "I had a wonderful time with Alice! We talked about various topics and really enjoyed each other's company. She seemed happy and we made plans to meet again soon."
```

## API Reference

- `POST /api/interactions/plan` - Plan new interaction
- `GET /api/interactions/planned/{companion_id}` - Get planned interactions
- `POST /api/interactions/{id}/complete` - Complete interaction with outcome
- `GET /api/interactions/history/{companion_id}/{third_party_id}` - Get history
- `POST /api/interactions/detect` - Detect interaction in message