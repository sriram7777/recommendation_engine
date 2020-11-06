
from mongo_factory.user_data_dao import UserData


class UserDataDaoImpl:

    @staticmethod
    def find_user(user_id):

        return UserData.objects.get(user_id=user_id)

    @staticmethod
    def insert_doc(doc):
        UserData.objects(user_id=doc.user_id).update(set__user_embedding=doc.user_embedding,
                                                     set__user_index=doc.user_index, upsert=True, multi=False)

