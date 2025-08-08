use std::process;

fn main() {
    println!("Testing Automatic Person Detection System");
    println!("==========================================");
    
    let test_messages = vec![
        "My friend Alice called me today",
        "I talked to my colleague Bob about the project", 
        "My boss Sarah wants to meet tomorrow",
        "Dr. Johnson is my new doctor",
        "I'm worried about my brother Mike",
        "John and I went to the store",
        "Mary said she likes the new restaurant",
        "I love spending time with Emma",
        "David is really smart and helpful",
        "My teacher Professor Smith assigned homework"
    ];
    
    println!("Test messages:");
    for (i, message) in test_messages.iter().enumerate() {
        println!("{}. {}", i + 1, message);
    }
    
    println!("\n✅ Test messages prepared. Build the system and test via API endpoints:");
    println!("   POST /api/persons/detect - Detect persons in a message");
    println!("   GET /api/persons - List all detected persons");
    println!("   GET /api/persons/{{name}} - Get specific person details");
}