# Automatic Person Detection System Tests

## Test Scenarios

### 1. Basic Name Detection
```bash
curl -X POST http://localhost:3000/api/persons/detect \
  -H "Content-Type: application/json" \
  -d '{"prompt": "My friend Alice called me today"}'
```

Expected: Should detect "Alice" and create third-party individual with relationship "friend"

### 2. Multiple Persons in One Message
```bash
curl -X POST http://localhost:3000/api/persons/detect \
  -H "Content-Type: application/json" \
  -d '{"prompt": "John and Mary went to see Dr. Smith"}'
```

Expected: Should detect "John", "Mary", and "Dr. Smith" with appropriate occupations

### 3. Relationship Context Detection
```bash
curl -X POST http://localhost:3000/api/persons/detect \
  -H "Content-Type: application/json" \
  -d '{"prompt": "My colleague Bob from work is really helpful"}'
```

Expected: Should detect "Bob" with relationship "colleague" and personality trait "helpful"

### 4. Emotional Context Detection
```bash
curl -X POST http://localhost:3000/api/persons/detect \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I love spending time with Emma, she makes me happy"}'
```

Expected: Should detect "Emma" with positive emotional valence affecting initial attitudes

### 5. Family Relationship Detection
```bash
curl -X POST http://localhost:3000/api/persons/detect \
  -H "Content-Type: application/json" \
  -d '{"prompt": "My brother Mike is worried about his job"}'
```

Expected: Should detect "Mike" with relationship "brother" and higher trust/empathy values

### 6. Professional Context
```bash
curl -X POST http://localhost:3000/api/persons/detect \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Professor Johnson teaches computer science"}'
```

Expected: Should detect "Johnson" with occupation "professor" and relationship "teacher"

### 7. Duplicate Detection
```bash
# First message
curl -X POST http://localhost:3000/api/persons/detect \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Alice is my friend"}'

# Second message with same person
curl -X POST http://localhost:3000/api/persons/detect \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Alice and I went shopping"}'
```

Expected: Should create Alice once, then increment mention_count on second detection

### 8. View All Detected Persons
```bash
curl -X GET http://localhost:3000/api/persons
```

Expected: Should return JSON array of all detected third-party individuals

### 9. Get Specific Person Details
```bash
curl -X GET http://localhost:3000/api/persons/Alice
```

Expected: Should return detailed information about Alice including attitudes

### 10. Integration with Message Processing
```bash
curl -X POST http://localhost:3000/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I had lunch with my coworker David today"}'
```

Expected: Should process the message normally AND automatically detect "David" in the background

## Success Criteria

1. ✅ Names are extracted correctly using NLP patterns
2. ✅ Relationship context is identified and stored
3. ✅ Occupation information is extracted when available
4. ✅ Personality traits are identified from context
5. ✅ Initial attitude values are set based on context
6. ✅ Duplicate detection prevents creating same person twice
7. ✅ Mention counts are updated for existing persons
8. ✅ Integration with existing message processing works seamlessly
9. ✅ API endpoints provide access to person data
10. ✅ System handles edge cases gracefully

## Architecture Summary

- **Person Detection**: Regex-based NLP patterns identify person names in messages
- **Context Analysis**: Extracts relationships, occupations, and personality traits  
- **Attitude Initialization**: Creates context-based initial 14-dimensional attitudes
- **Database Integration**: Stores in existing third_party_individuals table
- **API Integration**: Automatic detection in /api/prompt + dedicated person endpoints
- **Memory System**: Creates initial memories and tracks interactions