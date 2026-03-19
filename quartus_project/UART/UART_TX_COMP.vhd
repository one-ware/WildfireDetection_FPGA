-- filepath: one_ai/Quartus_IP/UART_TX.vhd
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.numeric_std.all;

entity UART_TX_COMP is
    generic (
        CLK_FREQ  : natural := 50_000_000;
        BAUD_RATE : natural := 115200
    );
    port (
        clk       : in std_logic;
        reset_n   : in std_logic;
        tx        : out std_logic;
        tx_data   : in std_logic_vector(7 downto 0);
        tx_start  : in std_logic;
        tx_new_frame  : in std_logic;
        tx_busy   : out std_logic
    );
end UART_TX_COMP;

architecture rtl of UART_TX_COMP is
    constant CLKS_PER_BIT : natural := CLK_FREQ / BAUD_RATE;
    constant BIT_TIMER_MAX : natural := CLKS_PER_BIT - 1;
    
    type state_type is (IDLE, START_BIT, DATA_BITS, STOP_BIT);
    signal state : state_type := IDLE;
    
    signal bit_timer : natural range 0 to BIT_TIMER_MAX := 0;
    signal bit_counter : natural range 0 to 7 := 0;
    signal tx_shift_reg : std_logic_vector(7 downto 0) := (others => '0');
	
	constant RB_SIZE : natural := 16387;
	type rb_type is array(0 to RB_SIZE) of std_logic_vector(7 downto 0);
	signal ring_buffer : rb_type;
	signal head : integer range rb_type'range;
	signal tail : integer range rb_type'range;
	
	-- mini state machine to push out tx-header
    signal new_frame_header : natural range 0 to 3 := 0;
    signal tx_data_reg : std_logic_vector(7 downto 0) := (others => '0');
    
begin

    process(clk)
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                state <= IDLE;
                bit_timer <= 0;
                bit_counter <= 0;
                tx_shift_reg <= (others => '0');
                tx <= '1';
                tx_busy <= '0';
				tail <= 0;
            else
                case state is
                    when IDLE =>
                        tx <= '1';
                        tx_busy <= '0';
                        bit_timer <= 0;
                        bit_counter <= 0;
                        if head /= tail then
                            tx_shift_reg <= ring_buffer(tail);
							if tail >= RB_SIZE then
								tail <= 0;
							else
								tail <= tail + 1;
							end if;
                            tx_busy <= '1';
                            state <= START_BIT;
                        end if;
                    
                    when START_BIT =>
                        tx <= '0'; -- Start bit
                        if bit_timer = BIT_TIMER_MAX then
                            bit_timer <= 0;
                            state <= DATA_BITS;
                        else
                            bit_timer <= bit_timer + 1;
                        end if;
                    
                    when DATA_BITS =>
                        tx <= tx_shift_reg(0); -- Send LSB first
                        if bit_timer = BIT_TIMER_MAX then
                            bit_timer <= 0;
                            tx_shift_reg <= '0' & tx_shift_reg(7 downto 1); -- Shift right
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
                        tx <= '1'; -- Stop bit
                        if bit_timer = BIT_TIMER_MAX then
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
		
	process(clk)
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                head <= 0;
				new_frame_header <= 0;
            else
				if tx_start = '1' or new_frame_header /= 0 then --and ((head +1) mod 16) /= tail 
					if tx_new_frame = '1' then
						ring_buffer(head) <= x"FF";
						new_frame_header <= 1;
						tx_data_reg <= tx_data;
					elsif new_frame_header = 1 then
						ring_buffer(head) <= x"AA";
						new_frame_header <= 2;
					elsif new_frame_header = 2 then
						ring_buffer(head) <= x"55";
						new_frame_header <= 3;
					elsif new_frame_header = 3 then
						ring_buffer(head) <= tx_data_reg;
						new_frame_header <= 0;
					else
						ring_buffer(head) <= tx_data;
					end if;
					
					if head >= RB_SIZE then
						head <= 0;
					else
						head <= head + 1;
					end if;
				end if;
            end if;
        end if;
    end process;

end rtl;