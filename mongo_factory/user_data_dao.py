from mongoengine import *
from datetime import datetime


class AbstractEntity(Document):
    enabled = BooleanField(default=True)
    updated_at = DateTimeField(db_field='updated_at', default=datetime.utcnow())

    meta = {
        'allow_inheritance': True,
        'abstract': True
    }


class UserData(AbstractEntity):
    user_id = StringField(max_length=50, primary_key=True)
    user_embedding = ListField(FloatField())
    user_index = IntField(null=False)

    meta = {
        'collection': 'user_data',
        'allow_inheritance': False
    }
