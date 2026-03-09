from main import app
from development.db import db, Category, Role, Position, LevelEnum


def seed_categories():

    categories = [
        "Software Development",
        "Data & Infrastruktur",
        "Product & Quality"
    ]

    for name in categories:
        exist = Category.query.filter_by(name=name).first()
        if not exist:
            db.session.add(Category(name=name))

    db.session.commit()
    print("✅ Categories seeded")


def seed_roles():

    software = Category.query.filter_by(name="Software Development").first()
    data = Category.query.filter_by(name="Data & Infrastruktur").first()
    product = Category.query.filter_by(name="Product & Quality").first()

    roles = [
        Role(name="Frontend Developer", category_id=software.id),
        Role(name="Backend Developer", category_id=software.id),
        Role(name="Software Engineer", category_id=software.id),

        Role(name="Data Scientist", category_id=data.id),
        Role(name="DevOps Engineer", category_id=data.id),

        Role(name="Product Manager", category_id=product.id),
        Role(name="UX Designer", category_id=product.id),
        Role(name="QA Engineer", category_id=product.id),
    ]

    for role in roles:
        exist = Role.query.filter_by(name=role.name).first()
        if not exist:
            db.session.add(role)

    db.session.commit()
    print("✅ Roles seeded")


def seed_positions():

    roles = Role.query.all()

    for role in roles:

        for level in LevelEnum:
            exist = Position.query.filter_by(
                role_id=role.id,
                level=level
            ).first()

            if not exist:
                db.session.add(
                    Position(
                        title=level.value,
                        role_id=role.id,
                        level=level
                    )
                )

    db.session.commit()
    print("✅ Positions seeded")


if __name__ == "__main__":
    with app.app_context():
        seed_categories()
        seed_roles()
        seed_positions()