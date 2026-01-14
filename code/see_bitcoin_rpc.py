from bitcoinrpc.authproxy import AuthServiceProxy

rpc = AuthServiceProxy("http://cp1user:CP1SecurePassword123!@127.0.0.1:8332")

def get_mempool_entry(txid):
    try:
        return rpc.getmempoolentry(txid)
    except:
        return None

def get_mempool_info():
    return rpc.getmempoolinfo()
