import os
import shutil

def delete_user(user_name, data_dir='./registered_users'):
    """
    删除已注册用户的特征文件和相关数据。

    Args:
        user_name (str): 要删除的用户的姓名或标识符。
        data_dir (str): 已注册用户的目录，默认为 './registered_users'。
    """
    user_dir = os.path.join(data_dir, user_name)

    # 检查该用户是否存在
    if not os.path.exists(user_dir):
        print(f"User {user_name} not found in the registered users directory.")
        return

    # 删除该用户的特征嵌入文件夹及所有文件
    try:
        shutil.rmtree(user_dir)  # 删除该用户文件夹及其内容
        print(f"User {user_name} has been successfully deleted.")
    except Exception as e:
        print(f"An error occurred while deleting the user {user_name}: {str(e)}")

if __name__ == "__main__":
    user_name = input("Enter the user name you want to delete: ")
    delete_user(user_name)
