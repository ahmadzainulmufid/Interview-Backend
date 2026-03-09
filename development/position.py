from flask import Blueprint, request, jsonify
from development.db import db, Category, Role, Position, LevelEnum

position_bp = Blueprint('position', __name__)

@position_bp.route("/categories", methods=["GET"])
def get_categories():
    categories = Category.query.all()

    result = [
        {
            "id": c.id,
            "name": c.name
        }
        for c in categories
    ]

    return jsonify(result), 200

@position_bp.route("/roles", methods=["GET"])
def get_roles():
    category_id = request.args.get("category_id")

    query = Role.query

    if category_id:
        query = query.filter_by(category_id=category_id)

    roles = query.all()

    result = [
        {
            "id": r.id,
            "name": r.name,
            "category_id": r.category_id
        }
        for r in roles
    ]

    return jsonify(result), 200

@position_bp.route("/positions", methods=["GET"])
def get_positions():
    role_id = request.args.get("role_id")

    query = Position.query

    if role_id:
        query = query.filter_by(role_id=role_id)

    positions = query.all()

    result = [
        {
            "id": p.id,
            "title": p.title,
            "role_id": p.role_id,
            "level": p.level.value,
            "description": p.description
        }
        for p in positions
    ]

    return jsonify(result), 200

@position_bp.route("/positions", methods=["POST"])
def create_position():
    data = request.get_json()

    try:
        position = Position(
            title=data["title"],
            role_id=data["role_id"],
            level=LevelEnum(data["level"]),
            description=data.get("description")
        )

        db.session.add(position)
        db.session.commit()

        return jsonify({"message": "Position created successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 400