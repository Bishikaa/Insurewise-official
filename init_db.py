import sqlite3
from werkzeug.security import generate_password_hash

conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)
''')

# Add a test user (email: test@example.com, password: test123)
hashed_password = generate_password_hash('test123')
c.execute('INSERT OR IGNORE INTO users (email, password) VALUES (?, ?)', ('test@example.com', hashed_password))

conn.commit()
conn.close()

print("Database initialized with test user.")
