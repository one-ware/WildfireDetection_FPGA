-- filepath: one_ai/Quartus_IP/UART_RX.vhd
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.numeric_std.all;

entity UART_RX_COMP is
    generic (
        CLK_FREQ  : natural := 50_000_000;
        BAUD_RATE : natural := 115200
    );
    port (
        clk       : in std_logic;
        reset_n   : in std_logic;
        rx        : in std_logic;
        rx_data   : out std_logic_vector(7 downto 0);
        rx_valid  : out std_logic
    );
end UART_RX_COMP;

architecture rtl of UART_RX_COMP is
    constant CLKS_PER_BIT : natural := CLK_FREQ / BAUD_RATE;
    constant BIT_TIMER_MAX : natural := CLKS_PER_BIT - 1;
    
    type state_type is (IDLE, START_BIT, DATA_BITS, STOP_BIT);
    signal state : state_type := IDLE;
    
    signal bit_timer : natural range 0 to BIT_TIMER_MAX := 0;
    signal bit_counter : natural range 0 to 7 := 0;
    signal rx_shift_reg : std_logic_vector(7 downto 0) := (others => '0');
    signal rx_sync : std_logic_vector(2 downto 0) := (others => '1');
    
begin

    -- Synchronize RX input
    process(clk)
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                rx_sync <= (others => '1');
            else
                rx_sync <= rx_sync(1 downto 0) & rx;
            end if;
        end if;
    end process;

    -- UART RX state machine
    process(clk)
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                state <= IDLE;
                bit_timer <= 0;
                bit_counter <= 0;
                rx_shift_reg <= (others => '0');
                rx_data <= (others => '0');
                rx_valid <= '0';
            else
                rx_valid <= '0'; -- Default
                
                case state is
                    when IDLE =>
                        bit_timer <= 0;
                        bit_counter <= 0;
                        if rx_sync(2) = '0' then -- Start bit detected
                            state <= START_BIT;
                        end if;
                    
                    when START_BIT =>
                        if bit_timer = BIT_TIMER_MAX/2 then -- Sample in middle of bit
                            if rx_sync(2) = '0' then -- Valid start bit
                                bit_timer <= 0;
                                state <= DATA_BITS;
                            else
                                state <= IDLE; -- False start bit
                            end if;
                        else
                            bit_timer <= bit_timer + 1;
                        end if;
                    
                    when DATA_BITS =>
                        if bit_timer = BIT_TIMER_MAX then
                            bit_timer <= 0;
                            rx_shift_reg <= rx_sync(2) & rx_shift_reg(7 downto 1); -- Shift in LSB first
                            if bit_counter = 7 then
                                bit_counter <= 0;
                                state <= STOP_BIT;
                            else
                                bit_counter <= bit_counter + 1;
                            end if;
                        else
                            bit_timer <= bit_timer + 1;
                        end if;
                    
                    when STOP_BIT =>
                        if bit_timer = BIT_TIMER_MAX then
                            if rx_sync(2) = '1' then -- Valid stop bit
                                rx_data <= rx_shift_reg;
                                rx_valid <= '1';
                            end if;
                            state <= IDLE;
                        else
                            bit_timer <= bit_timer + 1;
                        end if;
                    
                    when others =>
                        state <= IDLE;
                end case;
            end if;
        end if;
    end process;

end rtl;