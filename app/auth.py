"""
认证模块，用户登录验证+jwt令牌签发+接口鉴权
完整流程：用户提交账户密码->验证通过->签发jwt令牌
        ->后续请求带上令牌->自动验证令牌->提取当前用户
"""

# 导入依赖
# datetime：获取当前时间，用于计算令牌过期时间
# timedelta：表示时间差，如30分钟后过期
# timezone：utc时区，确保服务器在任何时区都生成一致的过期时间
from datetime import datetime,timedelta, timezone
# bcrypt:密码哈希库
# 不存储明文密码，只存储哈希值，验证时把用户输入的密码重新哈希后与存储对比
import bcrypt
# jwt(PyJWT):JSON Web Token编解码库
# 用于生成登录令牌（encode）和验证令牌（decode）
import jwt

# Depends:FastAPI的依赖注入，自动在接口执行前运行前置函数
from fastapi import Depends, HTTPException
# fastapi内置的oauth2密码模式工具
# 他会自动从请求的Authorization:Bearer<token>头中提取token字符串
from fastapi.security import OAuth2PasswordBearer

# jwt解码失败时抛出异常
# 包括签名错误、过期、格式损坏等所有情况
from jwt.exceptions import  InvalidTokenError

from app.config import settings


#演示用户数据（生产环境替换为数据库）
# 存储两个测试账号的密码哈希值
# bcrypt.hashpw(明文密码,随机盐)->生成不可逆的哈希值（哈希是一种单向计算：能从密码算出哈希值，但不能从哈希值反推出密码。）
# 每次启动服务时gensalt()会生成新的盐，所以哈希值每次不同，但验证仍然能通过
# 盐就是一段随机字符串，在哈希之前拼到密码后面，让同样的密码产生不同的哈希值。
# key= 用户名，value = 密码的bcrypt哈希值
# 如果不使用盐，两个用户使用同一个密码，对应的哈希值也一样
DEMO_USERS = {
    "alice":bcrypt.hashpw(b"alice-secret", bcrypt.gensalt()),
    "bob":bcrypt.hashpw(b"bob-secret",bcrypt.gensalt()),
}

# Oauth2令牌提取器
# 告诉fastapi：
# 登录接口地址是："auth/token"
# 后续受保护的接口，自动从请求头Authorization:Bearer xxx中提取token
# 它本身不做验证，只负责把token字符串提取出来
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


# 验证用户名和密码
# 在登录接口中调用，判断用户提交的账户密码是否正确
def authenticate_user(username: str, password: str) -> bool:
    # 把密码字符串编码为字符串，bcrypt只接受bytes类型
    password_bytes = password.encode("utf-8")

    # bcrypt有72个字节长度限制，超过的部分会被静默截断
    # 可能导致不同密码产生相同哈希，产生安全漏洞
    # 所以超长密码直接拒绝
    if len(password_bytes) > 72:
        return False

    # 从字典中取出该用户的密码哈希值
    # DEMO_USER["alice"]不是用户名，是alice的密码哈希值
    # 用username查字典，找到了返回对应的值，找不到返回default
    password_hash = DEMO_USERS.get(username,DEMO_USERS["alice"])
    # bcrypt.checkpw：把用户输入的密码重新哈希，与存储的哈希值对比
    # 内部会自动从password_hash中提取盐值，不需要单独传盐
    password_matches = bcrypt.checkpw(password_bytes, password_hash)
    # 两个条件都要满足：
    #   用户名确实存在DEMO_USERS中
    #   密码哈希比对通过
    # 注意即使用户名不存在，上面也做了一次checkpw，
    # 所以这里还需要username in DEMO_USERS来确认用户名是否真的存在
    return username in DEMO_USERS and password_matches


# 签发jwt令牌
# 验证通过后调用，生成一个带过期时间的令牌返回给前端
def create_access_token(username: str) -> str:
    # 计算过期时间= 当前时间 + 配置的有效分钟数
    expires_at = datetime.now(timezone.utc) + timedelta(
        # app/config.py 里定义的。文件开头导入了它
        minutes=settings.jwt_expire_minutes
    )
    # jwt的载荷，存放令牌内部的数据
    # sub(subject)：令牌的主题，这里存用户名，后续解码是用它识别这个令牌属于谁
    # exp(expiration)：过期时间，jwt库解码时会自动检查，过期则抛异常
    payload = {
        # payload 就是一个普通字典，装的是想塞进令牌里的信息：
        # 就像是写一张纸条：这张通行证属于username，有效期到expires_at
        "sub": username,
        "exp": expires_at,
    }
    """
    第一件事：把 payload 编码成一段乱码
    {"sub":"alice","exp":1759700189}
    → Base64 编码
    → eyJzdWIiOiJhbGljZSIsImV4cCI6MTc1OTcwMDE4OX0

    第二件事：用密钥生成签名（防篡改）
    用 "my-secret-key" 对上面的内容做 HMAC-SHA256 运算
    → 得到签名：SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c

    第三件事：拼接成最终的 JWT 字符串
    头部.载荷.签名
    eyJhbGci...  .  eyJzdWIi...  .  SflKxw...
    ↑ 说明用的算法    ↑ 你装的数据     ↑ 防篡改的签名
    """
    return jwt.encode(# 把纸条变成令牌
        payload,
        settings.jwt_secret,
        algorithm="HS256",
    )



# 从令牌中提取当前用户（接口鉴权）
# 作为Depends注入到受保护的接口中，每次请求自动执行
"""
用户发请求：GET /ask，Headers: Authorization: Bearer eyJhbG...
    │
    │  ① FastAPI 看到参数有 Depends(oauth2_scheme)
    │
    ▼
oauth2_scheme 执行：
    从请求头提取 "eyJhbG..." → 返回这个字符串
    │
    │  ② 返回值自动填入 token 参数
    │
    ▼
get_current_user(token="eyJhbG...") 执行：
    解码 token → 验证签名 → 检查过期 → 返回 "alice"
"""
def get_current_user(
        # Depends(oauth2_scheme) 的执行顺序：
        #   1. oauth2_scheme 从请求头中提取 Bearer token
        #   2. 提取到的 token 字符串自动传给 get_current_user 的 token 参数
        # 如果请求头中没有 Authorization 字段，FastAPI 直接返回 401。
        token: str = Depends(oauth2_scheme),
) -> str:

    # 预先创建错误响应对象，后面多处可能用到，避免重复写。
    # headers 中的 WWW-Authenticate: Bearer 告诉客户端"需要 Bearer 令牌认证"。
    credentials_error = HTTPException(
        status_code=401,
        detail="登录凭证无效或已过期，请重新登录",
        headers={"www-Authenticate":"Bearer"},
    )

    try:
        # jwt.decode: 解码并验证令牌。
        #   1. 用 jwt_secret 验证签名 → 签名不匹配说明令牌被篡改
        #   2. 检查 exp 字段 → 已过期则报错
        #   3. options={"require": ["sub","exp"]} → 强制要求 payload 中
        #      必须包含 sub 和 exp 字段，缺少则报错
        # 验证全部通过后返回解码后的 payload 字典。
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            options={"require":["sub","exp"]}
        )
    except InvalidTokenError:
        raise credentials_error

    # 从payload中取出用户名
    username = payload["sub"]
    # 双重校验：
    #   1. sub 的值必须是字符串（防止恶意构造 payload 传入数字或对象）
    #   2. 用户名必须在 DEMO_USERS 中（防止用一个已删除用户的旧令牌访问）
    if not isinstance(username, str) or username not in DEMO_USERS:
            raise credentials_error
    # 验证全部通过，返回用户名。
    # 受保护的接口函数就能拿到这个用户名：
    #
    #   @app.get("/profile")
    #   def profile(username: str = Depends(get_current_user)):
    #       return {"user": username}   ← 这里的 username 就是 "alice" 或 "bob"
    return username