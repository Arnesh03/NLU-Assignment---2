"""
==============================================================================
Dataset Generator — 1000 Unique Indian Full Names
==============================================================================
Generates Indian names by randomly pairing first names with common surnames.

The pool contains ~250 authentic Indian first names (male and female) and
~100 common Indian surnames spanning multiple regions and communities.
Random pairing produces 1000 unique full names for training.

Author: Arnesh Singh
Course: NLU Assignment 2
==============================================================================
"""

import random
import os

# ── Indian First Names ──────────────────────────────────────────────────────
# Curated mix of male and female first names from various Indian regions
FIRST_NAMES = [
    # Male names
    "Aarav", "Aadi", "Aadhya", "Aahana", "Aarush", "Aayush", "Abhinav", "Aditya", "Advait",
    "Agastya", "Ajay", "Akash", "Akshat", "Alok", "Aman", "Amar", "Amit", "Amol", "Anand",
    "Aniket", "Anirudh", "Ankit", "Ankur", "Anshul", "Arjun", "Arnav", "Arun", "Ashok",
    "Ashwin", "Atharv", "Ayaan", "Ayush", "Bharat", "Chirag", "Darsh", "Deepak", "Dev",
    "Devansh", "Dhruv", "Dinesh", "Gaurav", "Girish", "Gopal", "Hardik", "Harish", "Hemant",
    "Hitesh", "Ishaan", "Ishan", "Jatin", "Jay", "Kabir", "Karan", "Kartik", "Kiaan",
    "Krish", "Krishna", "Kunal", "Laksh", "Lakshay", "Madhav", "Manan", "Manish", "Mayank",
    "Mohit", "Mukesh", "Naman", "Naveen", "Neeraj", "Nikhil", "Nitin", "Ojas", "Om",
    "Pankaj", "Parth", "Piyush", "Pranav", "Praneel", "Pratyush", "Rahul", "Raj", "Rajat",
    "Rajesh", "Rakesh", "Ranbir", "Ranveer", "Ravi", "Reyansh", "Rishabh", "Rishi", "Rohit",
    "Rohan", "Rudra", "Sachin", "Sahil", "Samarth", "Sameer", "Sandeep", "Sarthak", "Saurabh",
    "Shaurya", "Shivam", "Shrey", "Siddharth", "Sunil", "Suresh", "Tanmay", "Tanuj", "Tarun",
    "Tejas", "Utkarsh", "Varun", "Vedant", "Veer", "Vihaan", "Vijay", "Vikas", "Vinay",
    "Viraj", "Vishal", "Vivaan", "Vivek", "Yash", "Yatin", "Yuvraj",
    # Female names
    "Aanya", "Aarini", "Aarna", "Aashi", "Aditi", "Ahana", "Aisha", "Akshara", "Amrita",
    "Anamika", "Ananya", "Anika", "Anita", "Anjali", "Ankita", "Anusha", "Anvi", "Anya",
    "Aradhana", "Archana", "Arya", "Avani", "Avni", "Bhavana", "Bhavika", "Chaitra",
    "Charvi", "Chitra", "Damini", "Deepa", "Deepika", "Devika", "Diya", "Divya", "Ekta",
    "Eshana", "Falguni", "Gauri", "Gayatri", "Geeta", "Gunjan", "Harini", "Harita", "Hema",
    "Ira", "Isha", "Ishani", "Ishita", "Janaki", "Jaya", "Jyoti", "Kajal", "Kamala",
    "Kavya", "Keya", "Kiara", "Komal", "Krisha", "Kritika", "Lakshmi", "Lavanya", "Leela",
    "Madhavi", "Mahika", "Manisha", "Maya", "Meera", "Megha", "Meghana", "Mitali", "Mohini",
    "Nandini", "Neelima", "Neha", "Nidhi", "Niharika", "Nikita", "Nisha", "Pallavi",
    "Pavani", "Pooja", "Prachi", "Pragya", "Prisha", "Priya", "Priyanka", "Radhika",
    "Rashmi", "Reva", "Ria", "Ridhi", "Riya", "Ruhi", "Saanvi", "Sakshi", "Sanaya",
    "Sanya", "Sara", "Saumya", "Sejal", "Shravya", "Shruti", "Simran", "Siya", "Sneha",
    "Sonali", "Suhani", "Swara", "Tanvi", "Tara", "Trisha", "Urmi", "Urvashi", "Vaishnavi",
    "Varsha", "Vedika", "Vidhi", "Vrinda", "Yamini", "Yashvi", "Zara"
]

# ── Indian Surnames ─────────────────────────────────────────────────────────
# Common surnames from across India: North, South, East, West, and Central
SURNAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Mehta", "Shah", "Joshi",
    "Mishra", "Pandey", "Tiwari", "Dubey", "Srivastava", "Agarwal", "Chauhan", "Yadav",
    "Reddy", "Nair", "Menon", "Pillai", "Iyer", "Rao", "Naidu", "Choudhary", "Thakur",
    "Bhat", "Kaul", "Kapoor", "Malhotra", "Chopra", "Arora", "Bhatia", "Khanna", "Sethi",
    "Grover", "Sinha", "Banerjee", "Mukherjee", "Chatterjee", "Ghosh", "Das", "Sengupta",
    "Bose", "Roy", "Dutta", "Majumdar", "Bhatt", "Desai", "Jain", "Saxena", "Trivedi",
    "Kulkarni", "Deshpande", "Patil", "Pawar", "Jadhav", "More", "Shinde", "Gaikwad",
    "Nayak", "Hegde", "Shetty", "Kamath", "Bhandari", "Pokharel", "Rathore", "Shekhawat",
    "Rajput", "Solanki", "Parmar", "Garg", "Mittal", "Goyal", "Bansal", "Singhal",
    "Khatri", "Mahajan", "Taneja", "Ahuja", "Bajaj", "Dhawan", "Lal", "Prasad", "Mohan",
    "Rajan", "Krishnan", "Subramaniam", "Natarajan", "Venkatesh", "Ramesh", "Ganesh",
    "Sundaram", "Chandra", "Narayan", "Shukla", "Dwivedi", "Upadhyay", "Ojha", "Dixit",
    "Awasthi"
]


def generate_names(num_names=1000, seed=42):
    """
    Generate unique Indian full names by randomly pairing first names with surnames.

    With ~250 first names × ~100 surnames, there are ~25,000 possible combinations,
    so reaching 1000 unique pairs is easy without excessive retries.

    Args:
        num_names: How many unique names to generate
        seed: Random seed for reproducibility

    Returns:
        Sorted list of unique full names
    """
    random.seed(seed)
    names = set()

    # Keep generating until we have enough unique names
    while len(names) < num_names:
        first = random.choice(FIRST_NAMES)
        last = random.choice(SURNAMES)
        names.add(f"{first} {last}")

    # Sort alphabetically for reproducibility and readability
    return sorted(names)[:num_names]


def main():
    """Generate the training dataset and save to TrainingNames.txt."""
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "TrainingNames.txt")
    names = generate_names(1000)

    with open(output_path, "w") as f:
        for name in names:
            f.write(name + "\n")

    print(f"  ✓ Generated {len(names)} unique Indian names → {output_path}")


if __name__ == "__main__":
    main()
