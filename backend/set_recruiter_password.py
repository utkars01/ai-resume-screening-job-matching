"""Set or reset a recruiter password after the reviewed schema update."""


from getpass import getpass

from sqlalchemy import text

from backend.app.database.session import engine
from backend.app.services.auth_security import hash_password


def main() -> int:
    email = input("Recruiter email: ").strip().lower()
    password = getpass("New password (12+ characters): ")
    confirmation = getpass("Confirm password: ")

    if password != confirmation:
        print("Passwords did not match.")
        return 1

    if not email:
        print("Recruiter email is required.")
        return 1

    if len(password) < 12:
        print("Password must contain at least 12 characters.")
        return 1


    password_hash = hash_password(password)
    with engine.begin() as connection:
        has_password_column = connection.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'users' AND column_name = 'password_hash'"
            )
        ).scalar()
        if not has_password_column:
            print("Apply the reviewed recruiter schema update first.")
            return 1

        updated = connection.execute(
            text(
                "UPDATE users SET password_hash = :password_hash  "
                "WHERE lower(email) = :email AND role = 'recruiter'"
            ),
            {"password_hash": password_hash, "email": email},
        )
        if updated.rowcount != 1:
            print("No unique recruiter account matched that email.")
            return 1

    print("Recruiter password updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
