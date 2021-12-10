from jaxnerf.db.db import db
def init():
    db.create_all()
