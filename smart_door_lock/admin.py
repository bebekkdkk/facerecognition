"""
Admin Utilities
- Manage users
- View database stats
- Export/import data
- Clear database
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import FaceDatabase


def print_menu():
    """Print admin menu"""
    menu = """
╔══════════════════════════════════════════╗
║   SMART DOOR LOCK - ADMIN UTILITIES      ║
╚══════════════════════════════════════════╝

1. View Database Statistics
2. List All Users
3. Delete User
4. Export User List
5. View Access Log
6. Clear Database
7. Exit

"""
    print(menu)


def view_stats(db):
    """View database statistics"""
    stats = db.get_stats()
    print("\n╔══════════════════════════════════╗")
    print("║        DATABASE STATISTICS       ║")
    print("╚══════════════════════════════════╝\n")
    print(f"Total Users:        {stats.get('total_users', 0)}")
    print(f"Total Embeddings:   {stats.get('total_embeddings', 0)}")
    print(f"\nEnrolled Users:")
    for user in stats.get('users', []):
        count = db.get_user_embeddings_count(user)
        print(f"  - {user}: {count} embeddings")
    print()


def list_users(db):
    """List all users"""
    users = db.get_all_users()
    print("\n╔══════════════════════════════════╗")
    print("║        ENROLLED USERS            ║")
    print("╚══════════════════════════════════╝\n")
    if users:
        for i, user in enumerate(users, 1):
            user_id = user.get('id', 'unknown')
            name = user.get('name', 'UNKNOWN')
            count = db.get_user_embeddings_count(user_id)
            print(f"{i}. {name} [{user_id}] ({count} samples)")
    else:
        print("No users enrolled.")
    print()


def delete_user(db):
    """Delete user dari database"""
    users = db.get_all_users()
    if not users:
        print("\n[INFO] No users to delete")
        return
    
    print("\n Available users:")
    for i, user in enumerate(users, 1):
        print(f"{i}. {user.get('name', 'UNKNOWN')} [{user.get('id', 'unknown')}]")
    
    try:
        choice = int(input("\nSelect user number to delete (0 to cancel): "))
        if choice == 0:
            return
        if 1 <= choice <= len(users):
            user = users[choice - 1]
            user_id = user.get('id')
            user_name = user.get('name', 'UNKNOWN')
            confirm = input(f"\nConfirm delete '{user_name}' [{user_id}]? (yes/no): ").lower()
            if confirm == 'yes':
                db.delete_user(user_id)
                print(f"[SUCCESS] User '{user_name}' deleted!")
            else:
                print("[CANCELLED]")
        else:
            print("[ERROR] Invalid selection")
    except ValueError:
        print("[ERROR] Invalid input")


def export_users(db):
    """Export user list to JSON"""
    from datetime import datetime
    stats = db.get_stats()
    users_data = {
        "timestamp": datetime.now().isoformat(),
        "total_users": stats.get('total_users', 0),
        "users": {}
    }
    
    for user in stats.get('users', []):
        count = db.get_user_embeddings_count(user)
        users_data['users'][user] = {
            "enrollment_count": count
        }
    
    filename = "user_list.json"
    try:
        with open(filename, 'w') as f:
            json.dump(users_data, f, indent=2)
        print(f"\n[SUCCESS] User list exported to: {filename}")
    except Exception as e:
        print(f"\n[ERROR] Export failed: {e}")


def view_access_log():
    """View access log"""
    from config import DATA_DIR
    log_file = os.path.join(DATA_DIR, 'access_log.json')
    
    if not os.path.exists(log_file):
        print("\n[INFO] No access log found")
        return
    
    try:
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        print("\n╔══════════════════════════════════╗")
        print("║        ACCESS LOG                ║")
        print("╚══════════════════════════════════╝\n")
        
        # Show last 20 entries
        for log in logs[-20:]:
            timestamp = log['timestamp']
            status = log['status']
            name = log['name']
            similarity = log.get('similarity', 'N/A')
            
            status_display = "✓ GRANTED" if status == "GRANTED" else "✗ DENIED"
            print(f"{timestamp} | {status_display:10} | {name:15} | {similarity}")
        
        print(f"\nTotal access attempts: {len(logs)}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to read log: {e}")


def clear_database(db):
    """Clear entire database"""
    confirm = input("\n[WARNING] This will delete ALL data!\nConfirm? (type 'YES' in uppercase): ")
    
    if confirm == 'YES':
        try:
            users = db.get_all_users()
            for user in users:
                db.delete_user(user.get('id'))
            print("[SUCCESS] Database cleared!")
        except Exception as e:
            print(f"[ERROR] Clear failed: {e}")
    else:
        print("[CANCELLED]")


def main():
    """Main admin interface"""
    
    db = FaceDatabase()
    
    while True:
        print_menu()
        
        try:
            choice = input("Select option: ").strip()
            
            if choice == '1':
                view_stats(db)
            elif choice == '2':
                list_users(db)
            elif choice == '3':
                delete_user(db)
            elif choice == '4':
                export_users(db)
            elif choice == '5':
                view_access_log()
            elif choice == '6':
                clear_database(db)
            elif choice == '7':
                print("[INFO] Exiting admin panel")
                break
            else:
                print("[ERROR] Invalid option!")
                
        except KeyboardInterrupt:
            print("\n[ABORT] Exiting...")
            break
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
