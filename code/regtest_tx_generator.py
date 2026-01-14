#!/usr/bin/env python
"""
Improved Regtest Transaction Generator
======================================
Generates unique, valid transactions for testing CP1.
Each transaction uses fresh UTXOs to avoid duplicate rejections.
"""

import os
import time
import random
import logging
from typing import Optional, List
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("regtest_generator")


@dataclass
class GeneratedTx:
    """Generated transaction info."""
    txid: str
    size: int
    fee: float
    inputs: int
    outputs: int


class RegtestTxGenerator:
    """
    Generates unique transactions for regtest testing.
    
    Each transaction:
    - Spends a fresh UTXO (avoids duplicate rejection)
    - Has unique outputs
    - Optionally includes OP_RETURN with random nonce
    
    Usage:
        gen = RegtestTxGenerator(rpc)
        
        # Ensure funded
        gen.setup()
        
        # Generate transactions
        for tx in gen.generate_batch(count=10):
            print(f"Generated: {tx.txid}")
    """
    
    def __init__(self, rpc, wallet_name: str = "testwallet"):
        """
        Initialize generator.
        
        Args:
            rpc: Bitcoin RPC client
            wallet_name: Wallet to use for transactions
        """
        self.rpc = rpc
        self.wallet_name = wallet_name
        self._setup_complete = False
    
    def setup(self, initial_blocks: int = 110):
        """
        Setup regtest environment with funded wallet.
        
        Args:
            initial_blocks: Blocks to mine for initial funding
        """
        logger.info("Setting up regtest environment...")
        
        # Create or load wallet
        try:
            self.rpc.createwallet(self.wallet_name)
            logger.info(f"Created wallet: {self.wallet_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                try:
                    self.rpc.loadwallet(self.wallet_name)
                except:
                    pass  # Already loaded
            else:
                logger.warning(f"Wallet setup: {e}")
        
        # Generate initial blocks for funding
        addr = self.rpc.getnewaddress()
        blocks = self.rpc.generatetoaddress(initial_blocks, addr)
        logger.info(f"Mined {len(blocks)} blocks to {addr}")
        
        # Check balance
        balance = self.rpc.getbalance()
        logger.info(f"Wallet balance: {balance} BTC")
        
        self._setup_complete = True
    
    def get_fresh_utxo(self, min_amount: float = 0.001) -> Optional[dict]:
        """
        Get a fresh unspent UTXO.
        
        Args:
            min_amount: Minimum UTXO value
            
        Returns:
            UTXO dict or None if none available
        """
        utxos = self.rpc.listunspent(1, 9999999)
        
        # Filter by amount and shuffle for randomness
        valid = [u for u in utxos if u["amount"] >= min_amount]
        
        if not valid:
            return None
        
        return random.choice(valid)
    
    def generate_single(
        self, 
        include_op_return: bool = True,
        random_outputs: int = 2
    ) -> Optional[GeneratedTx]:
        """
        Generate a single unique transaction.
        
        Args:
            include_op_return: Include random OP_RETURN data
            random_outputs: Number of random outputs
            
        Returns:
            GeneratedTx or None if failed
        """
        # Get fresh UTXO
        utxo = self.get_fresh_utxo()
        if not utxo:
            # Need more funding - mine a block
            addr = self.rpc.getnewaddress()
            self.rpc.generatetoaddress(1, addr)
            utxo = self.get_fresh_utxo()
            if not utxo:
                logger.error("No UTXOs available")
                return None
        
        # Build inputs
        inputs = [{
            "txid": utxo["txid"],
            "vout": utxo["vout"]
        }]
        
        # Calculate amounts
        total_in = float(utxo["amount"])
        fee = 0.0001
        output_total = total_in - fee
        
        # Build outputs
        outputs = {}
        
        # Random destination outputs
        per_output = round(output_total / (random_outputs + 1), 8)
        for _ in range(random_outputs):
            addr = self.rpc.getnewaddress()
            outputs[addr] = per_output
            output_total -= per_output
        
        # Final output with remainder
        final_addr = self.rpc.getnewaddress()
        outputs[final_addr] = round(output_total, 8)
        
        # Add OP_RETURN with random nonce
        if include_op_return:
            nonce = random.getrandbits(64)
            outputs["data"] = format(nonce, "016x")
        
        try:
            # Create raw transaction
            raw = self.rpc.createrawtransaction(inputs, outputs)
            
            # Sign
            signed = self.rpc.signrawtransactionwithwallet(raw)
            if not signed.get("complete"):
                logger.error("Signing incomplete")
                return None
            
            # Broadcast
            txid = self.rpc.sendrawtransaction(signed["hex"])
            
            # Get size
            decoded = self.rpc.decoderawtransaction(signed["hex"])
            size = decoded.get("vsize", len(signed["hex"]) // 2)
            
            return GeneratedTx(
                txid=txid,
                size=size,
                fee=fee,
                inputs=len(inputs),
                outputs=len(outputs) - (1 if include_op_return else 0)
            )
            
        except Exception as e:
            logger.error(f"Failed to create tx: {e}")
            return None
    
    def generate_batch(
        self,
        count: int = 10,
        delay_ms: int = 200,
        mine_every: int = 5
    ):
        """
        Generate a batch of unique transactions.
        
        Args:
            count: Number of transactions to generate
            delay_ms: Delay between transactions in ms
            mine_every: Mine a block every N transactions
            
        Yields:
            GeneratedTx for each successful transaction
        """
        for i in range(count):
            tx = self.generate_single()
            
            if tx:
                yield tx
                logger.info(f"[{i+1}/{count}] {tx.txid[:16]}... size={tx.size}")
            else:
                logger.warning(f"[{i+1}/{count}] Failed")
            
            # Optionally mine a block
            if mine_every and (i + 1) % mine_every == 0:
                addr = self.rpc.getnewaddress()
                self.rpc.generatetoaddress(1, addr)
                logger.debug("Mined confirmation block")
            
            # Delay
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)


def main():
    """Main entry point for CLI usage."""
    from bitcoinrpc.authproxy import AuthServiceProxy
    
    rpc_url = os.environ.get(
        "BITCOIN_RPC_URL",
        "http://cp1user:CP1SecurePassword123!@127.0.0.1:18443"
    )
    
    rpc = AuthServiceProxy(rpc_url)
    
    # Check regtest
    info = rpc.getblockchaininfo()
    if info["chain"] != "regtest":
        logger.error(f"Not regtest: {info['chain']}")
        return
    
    gen = RegtestTxGenerator(rpc)
    
    # Setup if needed
    if info["blocks"] < 110:
        gen.setup()
    
    # Generate transactions
    count = int(os.environ.get("TX_COUNT", "100"))
    
    logger.info(f"Generating {count} unique transactions...")
    
    generated = 0
    for tx in gen.generate_batch(count=count):
        generated += 1
    
    logger.info(f"Complete: {generated}/{count} transactions generated")


if __name__ == "__main__":
    main()
