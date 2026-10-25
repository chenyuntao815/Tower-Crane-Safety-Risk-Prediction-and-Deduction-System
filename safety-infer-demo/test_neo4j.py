# test_neo4j.py
import os
import sys
import traceback
from neo4j import GraphDatabase
from neo4j.exceptions import (
    AuthError,
    ServiceUnavailable,
    ConfigurationError,
    Neo4jError
)

def main():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    print("==== Neo4j 连接参数（脱敏） ====")
    print(f"URI      : {uri}")
    print(f"USER     : {user}")
    print(f"DATABASE : {database}")
    print(f"PWD_LEN  : {len(password)}")
    print("================================")

    if not password:
        print("[FAIL] NEO4J_PASSWORD 为空。")
        sys.exit(2)

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))

        # 连通性（含认证）
        driver.verify_connectivity()
        print("[OK] verify_connectivity 通过：网络/端口/认证可用。")

        # 用最兼容的查询，不用 currentUser/currentDatabase
        with driver.session(database=database) as session:
            rec = session.run("RETURN 1 AS ok").single()
            print(f"[OK] 查询成功，ok={rec['ok']}")
            print("结论：连接与认证正常。")

        sys.exit(0)

    except AuthError as e:
        print("[FAIL] 认证失败（用户名或密码错误）。")
        print(f"       详情: {e}")
        sys.exit(4)
    except ServiceUnavailable as e:
        print("[FAIL] 服务不可用（Neo4j未启动/端口错误）。")
        print(f"       详情: {e}")
        sys.exit(3)
    except ConfigurationError as e:
        print("[FAIL] 配置错误（URI格式等）。")
        print(f"       详情: {e}")
        sys.exit(6)
    except Neo4jError as e:
        print("[FAIL] Neo4j返回错误。")
        print(f"       类型: {type(e).__name__}")
        print(f"       详情: {e}")
        sys.exit(5)
    except Exception as e:
        print("[FAIL] 未知异常。")
        print(f"       类型: {type(e).__name__}")
        print(f"       详情: {e}")
        traceback.print_exc()
        sys.exit(9)
    finally:
        if driver is not None:
            driver.close()

if __name__ == "__main__":
    main()
