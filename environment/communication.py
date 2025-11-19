"""
Communication Module
V2X communication model based on 3GPP TR 37.885 standard
"""

import numpy as np


class Communication:
    """
    V2X Communication Class
    Implements communication model based on 3GPP TR 37.885 standard
    """
    
    def __init__(self, bandwidth=10, carrier_freq=5905, 
                 environment="urban", channel_condition="LOS"):

        self.environment = environment
        self.channel_condition = channel_condition
        self.bandwidth = bandwidth * 1e6  # Convert to Hz
        self.carrier_freq = carrier_freq  # MHz
        self.carrier_freq_ghz = carrier_freq / 1000  # GHz
        
        # Shadow fading standard deviation from 3GPP TR 37.885
        # Urban scenario
        self.urban_los_shadow_std = 4.0
        self.urban_nlosv_shadow_std = 6.0
        self.urban_nlos_shadow_std = 8.0
        # Highway scenario
        self.highway_los_shadow_std = 3.0
        self.highway_nlosv_shadow_std = 5.0
    
    def _calculate_path_loss(self, distance):

        d = max(1.0, distance)  # Minimum distance 1m
        
        if self.environment == "urban":
            if self.channel_condition == "LOS":
                path_loss = 38.77 + 16.7 * np.log10(d) + 18.2 * np.log10(self.carrier_freq_ghz)
            elif self.channel_condition == "NLOSv":
                path_loss = 36.85 + 30 * np.log10(d) + 18.9 * np.log10(self.carrier_freq_ghz)
            elif self.channel_condition == "NLOS":
                pl_los = 38.77 + 16.7 * np.log10(d) + 18.2 * np.log10(self.carrier_freq_ghz)
                pl_nlos = 36.85 + 40 * np.log10(d) + 18.9 * np.log10(self.carrier_freq_ghz)
                path_loss = max(pl_los, pl_nlos)
            else:
                path_loss = 38.77 + 16.7 * np.log10(d) + 18.2 * np.log10(self.carrier_freq_ghz)
        else:  # highway
            if self.channel_condition == "LOS":
                path_loss = 32.4 + 20 * np.log10(d) + 20 * np.log10(self.carrier_freq_ghz)
            elif self.channel_condition == "NLOSv":
                path_loss = 32.4 + 25.8 * np.log10(d) + 20 * np.log10(self.carrier_freq_ghz)
            else:
                path_loss = 32.4 + 20 * np.log10(d) + 20 * np.log10(self.carrier_freq_ghz)
        
        return path_loss
    
    def _calculate_shadow_fading(self):

        if self.environment == "urban":
            if self.channel_condition == "LOS":
                shadow_std = self.urban_los_shadow_std
            elif self.channel_condition == "NLOSv":
                shadow_std = self.urban_nlosv_shadow_std
            elif self.channel_condition == "NLOS":
                shadow_std = self.urban_nlos_shadow_std
            else:
                shadow_std = self.urban_los_shadow_std
        else:  # highway
            if self.channel_condition == "LOS":
                shadow_std = self.highway_los_shadow_std
            elif self.channel_condition == "NLOSv":
                shadow_std = self.highway_nlosv_shadow_std
            else:
                shadow_std = self.highway_los_shadow_std
        
        shadow_fading = np.random.normal(0, shadow_std)
        return shadow_fading
    
    def _calculate_small_scale_fading(self):

        x = np.random.normal(0, 1)
        y = np.random.normal(0, 1)
        
        if self.environment == "urban" and (self.channel_condition == "NLOSv" or 
                                           self.channel_condition == "NLOS"):
            # Urban NLOS scenario uses Rayleigh fading
            fading = np.sqrt(x**2 + y**2)
        else:
            # Other scenarios use Rician fading
            if self.environment == "urban":
                if self.channel_condition == "LOS":
                    k_factor_db = 6.0
                else:
                    k_factor_db = 0.0
            else:  # highway
                if self.channel_condition == "LOS":
                    k_factor_db = 8.0
                elif self.channel_condition == "NLOSv":
                    k_factor_db = 3.0
                else:
                    k_factor_db = 0.0
            
            k_linear = 10**(k_factor_db/10)
            a = np.sqrt(k_linear)
            fading = np.sqrt((x + a)**2 + y**2)
        
        return fading
    
    def _calculate_channel_gain(self, distance):

        path_loss_db = self._calculate_path_loss(distance)
        shadow_fading_db = self._calculate_shadow_fading()
        small_scale_fading = self._calculate_small_scale_fading()
        
        path_loss_linear = 10**(-path_loss_db/10)
        shadow_fading_linear = 10**(shadow_fading_db/10)
        
        total_gain = path_loss_linear * shadow_fading_linear * small_scale_fading
        
        return total_gain
    
    def _calculate_snr(self, channel_gain, noise_power=-174):

        transmit_power = 23  # dBm (typical base station transmit power)
        
        received_power_db = transmit_power + 10 * np.log10(channel_gain)
        total_noise_power_db = noise_power + 10 * np.log10(self.bandwidth)
        snr_db = received_power_db - total_noise_power_db
        snr_linear = 10**(snr_db/10)
        
        return snr_linear
    
    def calculate_transmission_time(self, task_size, vehicle, rsu):

        if not rsu.is_vehicle_in_coverage(vehicle.position):
            return None
        
        distance = abs(vehicle.position - rsu.position)
        channel_gain = self._calculate_channel_gain(distance)
        snr = self._calculate_snr(channel_gain)
        
        # Calculate channel capacity using Shannon formula
        channel_capacity = self.bandwidth * np.log2(1 + snr)
        
        # Adjust efficiency based on channel condition
        if self.channel_condition == "LOS":
            efficiency_factor = 0.85
        elif self.channel_condition == "NLOSv":
            efficiency_factor = 0.75
        else:  # NLOS
            efficiency_factor = 0.65
        
        actual_rate_bits = channel_capacity * efficiency_factor
        actual_rate_bytes = actual_rate_bits / 8
        actual_rate_bytes = max(actual_rate_bytes, 1024)  # Minimum 1KB/s
        
        transmission_time = task_size / actual_rate_bytes
        
        return transmission_time

