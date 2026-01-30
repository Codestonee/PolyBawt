"""
Script to manage token approvals for Polymarket trading.

Checks and grants ERC20 allowances for USDC and CTF tokens to:
1. CTF Exchange (Standard markets)
2. Neg Risk Exchange (Negative risk markets)
3. Neg Risk Adapter (Adapter for neg risk)

Based on official Polymarket examples.
"""

import os
import sys
import time
from typing import Optional

from web3 import Web3
try:
    from web3.middleware import ExtraDataToPOAMiddleware  # Web3 v7+
    geth_poa_middleware = ExtraDataToPOAMiddleware
except ImportError:
    from web3.middleware import geth_poa_middleware  # Web3 v6
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TokenManager:
    """Manages ERC20 token approvals."""
    
    # Contract Addresses (Polygon Mainnet)
    USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    
    # Exchange Addresses to Approve
    EXCHANGE_ADDRESSES = {
        "CTF Exchange": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
        "Neg Risk Exchange": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
        "Neg Risk Adapter": "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
    }
    
    # ABIs
    ERC20_ABI = [
        {
            "constant": True,
            "inputs": [
                {"name": "_owner", "type": "address"},
                {"name": "_spender", "type": "address"}
            ],
            "name": "allowance",
            "outputs": [{"name": "", "type": "uint256"}],
            "payable": False,
            "stateMutability": "view",
            "type": "function"
        },
        {
            "constant": False,
            "inputs": [
                {"name": "_spender", "type": "address"},
                {"name": "_value", "type": "uint256"}
            ],
            "name": "approve",
            "outputs": [{"name": "", "type": "bool"}],
            "payable": False,
            "stateMutability": "nonpayable",
            "type": "function"
        }
    ]
    
    ERC1155_ABI = [
        {
            "inputs": [
                {"internalType": "address", "name": "account", "type": "address"},
                {"internalType": "address", "name": "operator", "type": "address"}
            ],
            "name": "isApprovedForAll",
            "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [
                {"internalType": "address", "name": "operator", "type": "address"},
                {"internalType": "bool", "name": "approved", "type": "bool"}
            ],
            "name": "setApprovalForAll",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        }
    ]
    
    def __init__(self):
        """Initialize Web3 and contracts."""
        self.rpc_url = os.getenv("WEB3_RPC_URI", "https://polygon.drpc.org")
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Verify connection
        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to Polygon RPC")
        
        # Load private key
        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
        if not self.private_key:
            raise ValueError("POLYMARKET_PRIVATE_KEY not set")
            
        # Initialize account (Signer)
        self.account = self.w3.eth.account.from_key(self.private_key)
        self.signer_address = self.account.address
        print(f"🔹 Signer Wallet: {self.signer_address}")
        
        # Determine Funder Address (Proxy Wallet)
        self.funder_address = os.getenv("POLYMARKET_FUNDER_ADDRESS")
        if not self.funder_address:
            print("⚠️  POLYMARKET_FUNDER_ADDRESS not set. Checking Signer wallet.")
            self.check_address = self.signer_address
        else:
            # Validate checksum
            if self.w3.is_checksum_address(self.funder_address):
                self.check_address = self.funder_address
            else:
                self.check_address = self.w3.to_checksum_address(self.funder_address)
            print(f"🔹 Proxy Wallet (Funder): {self.check_address}")
        
        # Contracts
        self.usdc = self.w3.eth.contract(address=self.USDC_ADDRESS, abi=self.ERC20_ABI)
        self.ctf = self.w3.eth.contract(address=self.CTF_ADDRESS, abi=self.ERC1155_ABI)
        
        # Max uint256
        self.MAX_UINT = 2**256 - 1

    def check_approvals(self) -> dict:
        """Check status of all required approvals."""
        status = {}
        
        print(f"\n🔍 Checking Approvals for {self.check_address}...")
        for name, spender in self.EXCHANGE_ADDRESSES.items():
            # Check USDC Allowance
            usdc_allowance = self.usdc.functions.allowance(self.check_address, spender).call()
            usdc_ok = usdc_allowance > 1_000_000_000  # > 1000 USDC
            
            # Check CTF Approval
            ctf_approved = self.ctf.functions.isApprovedForAll(self.check_address, spender).call()
            
            status[name] = {
                "usdc": usdc_ok,
                "ctf": ctf_approved,
                "spender": spender
            }
            
            print(f"  • {name}:")
            print(f"    - USDC: {'✅' if usdc_ok else '❌'} (Allowance: {usdc_allowance / 1e6:.2f})")
            print(f"    - CTF:  {'✅' if ctf_approved else '❌'}")
            
        return status

    def grant_approvals(self, status: dict):
        """Grant missing approvals."""
        if self.check_address != self.signer_address:
            print("\n⚠️  CANNOT APPROVE FOR PROXY WALLET DIRECTLY")
            print("   You are using a Proxy Wallet. The Signer (EOA) cannot directly call 'approve' on USDC.")
            print("   Approvals for Proxy Wallets must be done via the Polymarket UI or Gnosis Safe.")
            print("\n   👉 Action: Go to Polymarket.com, make a small trade or enable trading to trigger approvals.")
            return

        print("\n📝 Granting Missing Approvals for Signer...")
        
        nonce = self.w3.eth.get_transaction_count(self.signer_address)
        
        for name, data in status.items():
            spender = data["spender"]
            
            # Approve USDC
            if not data["usdc"]:
                print(f"  • Approving USDC for {name}...")
                tx = self.usdc.functions.approve(spender, self.MAX_UINT).build_transaction({
                    'chainId': 137,
                    'gas': 100000,
                    'maxFeePerGas': self.w3.to_wei('50', 'gwei'),
                    'maxPriorityFeePerGas': self.w3.to_wei('30', 'gwei'),
                    'nonce': nonce,
                })
                self._send_tx(tx)
                nonce += 1
                time.sleep(2)  # Wait between txs
            
            # Approve CTF
            if not data["ctf"]:
                print(f"  • Approving CTF for {name}...")
                tx = self.ctf.functions.setApprovalForAll(spender, True).build_transaction({
                    'chainId': 137,
                    'gas': 100000,
                    'maxFeePerGas': self.w3.to_wei('50', 'gwei'),
                    'maxPriorityFeePerGas': self.w3.to_wei('30', 'gwei'),
                    'nonce': nonce,
                })
                self._send_tx(tx)
                nonce += 1
                time.sleep(2)

    def _send_tx(self, tx_dict):
        """Sign and send transaction."""
        signed_tx = self.w3.eth.account.sign_transaction(tx_dict, self.private_key)
        try:
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            print(f"    ➜ Tx Sent: {tx_hash.hex()}")
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt['status'] == 1:
                print("    ✅ Confirmed")
            else:
                print("    ❌ FAILED")
        except Exception as e:
            print(f"    ❌ Error: {e}")


if __name__ == "__main__":
    try:
        manager = TokenManager()
        status = manager.check_approvals()
        
        # Check if anything needs approval
        needs_approval = any(not (d["usdc"] and d["ctf"]) for d in status.values())
        
        if needs_approval:
            response = input("\nGrant missing approvals? This costs MATIC. (y/n): ")
            if response.lower() == 'y':
                manager.grant_approvals(status)
                print("\n✨ All done! Re-checking...")
                manager.check_approvals()
            else:
                print("\n⚠️  Trading may fail without approvals.")
        else:
            print("\n✅ All systems ready for trading!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
