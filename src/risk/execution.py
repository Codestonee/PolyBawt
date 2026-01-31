class DynamicQuoteShader:
    """
    Adjust bid-ask quotes based on market toxicity
    """
    def __init__(self, mid_price):
        self.mid_price = mid_price
        self.base_spread = 0.02
        self.inventory = 0
    
    def calculate_optimal_quotes(self, vpin, inventory, time_to_expiry):
        # 1. TOXICITY ADJUSTMENT
        toxicity_multiplier = 1 + (2 * vpin)
        
        # 2. INVENTORY ADJUSTMENT
        inventory_skew = inventory * 0.01
        
        adjusted_spread = self.base_spread * toxicity_multiplier
        adjusted_mid = self.mid_price + inventory_skew
        
        bid = adjusted_mid - adjusted_spread / 2
        ask = adjusted_mid + adjusted_spread / 2
        
        bid = min(bid, ask - 0.001)
        return bid, ask
