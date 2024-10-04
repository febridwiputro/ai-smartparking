from passlib.context import CryptContext
import argparse
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)




if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("-p", "--password", help="Password", required=True)
    args = vars(arg.parse_args())
    print(get_password_hash(args["password"]))
