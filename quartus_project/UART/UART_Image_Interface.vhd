-- filepath: one_ai/Quartus_IP/UART_Image_Interface.vhd
-- =============================================================================
-- UART Image Interface for ONEAI CNN Integration
-- =============================================================================
-- This module receives 128x128 RGB image data via UART and provides it to
-- the CNN system using the same streaming interface format.
--
-- Protocol:
-- - Baud rate: 115200 (configurable)
-- - Data format: 8-bit RGB values per pixel
-- - Transmission order: R0,G0,B0,R1,G1,B1,R2,G2,B2...
-- - Frame start: 0xFF 0xAA 0x55 (3 byte header)
-- - Frame size: 128*128*3 = 49,152 bytes + 3 header bytes = 49,155 total
-- =============================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.numeric_std.all;
use work.CNN_Config_Package.all;
use work.CNN_Data_Package.all;

entity UART_Image_Interface is
    generic (
        CLK_FREQ    : natural := 100_000_000;  -- System clock frequency in Hz
        BAUD_RATE   : natural := 3000000;--115200;      -- UART baud rate
        FRAME_WIDTH : natural := 128;         -- Image width
        FRAME_HEIGHT: natural := 128          -- Image height
    );
    port (
        clk         : in std_logic;
        reset_n     : in std_logic;
        
        -- UART interface
        uart_rx     : in std_logic;
        uart_tx     : out std_logic;
        
        -- CNN streaming interface (output to CNN)
        rxStream  : out CNN_Stream_T;
        rxData    : out CNN_Values_T(2 downto 0);
        
        -- CNN streaming interface (input from CNN)
        txStream  : in CNN_Stream_T;
        txData    : in CNN_Values_T(0 downto 0);
        
        -- Status outputs
        frame_ready : out std_logic;
        error_flag  : out std_logic;
        
        -- Debug/Status
        bytes_received : out std_logic_vector(15 downto 0);
        current_state  : out std_logic_vector(3 downto 0);
		
        heartbeat     : in std_logic
    );
end UART_Image_Interface;


architecture rtl of UART_Image_Interface is

    -- UART component
    component UART_RX_COMP is
        generic (
            CLK_FREQ  : natural := CLK_FREQ;
            BAUD_RATE : natural := BAUD_RATE
        );
        port (
            clk       : in std_logic;
            reset_n   : in std_logic;
            rx        : in std_logic;
            rx_data   : out std_logic_vector(7 downto 0);
            rx_valid  : out std_logic
        );
    end component;

    component UART_TX_COMP is
        generic (
            CLK_FREQ  : natural := CLK_FREQ;
            BAUD_RATE : natural := BAUD_RATE
        );
        port (
            clk       : in std_logic;
            reset_n   : in std_logic;
            tx        : out std_logic;
            tx_data   : in std_logic_vector(7 downto 0);
            tx_start  : in std_logic;
			tx_new_frame : in std_logic;
            tx_busy   : out std_logic
        );
    end component;

    -- Constants
    constant FRAME_SIZE : natural := FRAME_WIDTH * FRAME_HEIGHT;
    constant TOTAL_BYTES : natural := FRAME_SIZE; -- RGB
    constant HEADER_BYTE_1 : std_logic_vector(7 downto 0) := x"FF";
    constant HEADER_BYTE_2 : std_logic_vector(7 downto 0) := x"AA";
    constant HEADER_BYTE_3 : std_logic_vector(7 downto 0) := x"55";
    constant ECHO_BYTE : std_logic_vector(7 downto 0) := x"24";
    
    -- State machine
    type state_type is (
        IDLE_STATE,
        WAIT_HEADER_1,
        WAIT_HEADER_2,
        WAIT_HEADER_3,
        RECEIVE_PIXEL_DATA,
        OUTPUT_PIXEL,
        FRAME_COMPLETE,
        ERROR_STATE
    );
    signal state : state_type := IDLE_STATE;
    
    -- UART signals
    signal rx_data : std_logic_vector(7 downto 0);
    signal rx_valid : std_logic;
    signal tx_data : std_logic_vector(7 downto 0);
    signal tx_start : std_logic := '0';
    signal tx_new_frame : std_logic := '0';
    signal tx_busy : std_logic;
    
    -- Image buffer
    signal pixel_buffer : std_logic_vector(7 downto 0) := (others => '0');
    
    -- Counters and position tracking
    signal byte_counter : natural range 0 to TOTAL_BYTES := 0;
    signal pixel_counter : natural range 0 to FRAME_SIZE := 0;
    signal rgb_counter : natural range 0 to 2 := 0; -- 0=R, 1=G, 2=B
    signal output_pixel_counter : natural range 0 to FRAME_SIZE := 0;
    
    -- Pixel position calculation
    signal current_column : natural range 0 to FRAME_WIDTH-1 := 0;
    signal current_row : natural range 0 to FRAME_HEIGHT-1 := 0;
    signal last_column : natural range 0 to FRAME_WIDTH-1 := 0;
    signal last_row : natural range 0 to FRAME_HEIGHT-1 := 0;
    
    -- Current pixel data
    signal current_pixel_r : std_logic_vector(7 downto 0) := (others => '0');
    signal current_pixel_g : std_logic_vector(7 downto 0) := (others => '0');
    signal current_pixel_b : std_logic_vector(7 downto 0) := (others => '0');
    
    -- Output control
    signal output_active : std_logic := '0';
    signal output_delay_counter : natural range 0 to 112-5 := 0; -- Match ONE_AI_Interface timing
    constant OUTPUT_DELAY_CYCLES : natural := 112-5;
    signal rxStream_Reg  : CNN_Stream_t;
    signal rxData_Reg    : CNN_Values_T(2 downto 0);
    signal pixel_ready : std_logic := '0';
    
    -- Status signals
    signal frame_ready_int : std_logic := '0';
    signal error_flag_int : std_logic := '0';
    signal header_error_count : natural range 0 to 15 := 0;
	signal echo : std_logic := '0';
	signal beat : std_logic := '0';
	signal last_heart_beat : std_logic := '0';
    signal beat_count : natural range 0 to 15 := 0;
    signal debug_buf  : std_logic_vector(3 downto 0);
	
	-- Test rubish stuff
	signal uart_tx_rubish : std_logic := '0';
	signal debug_state : std_logic_vector(7 downto 0);

begin

    -- UART RX instance
    uart_rx_inst : UART_RX_COMP
    generic map (
        CLK_FREQ  => CLK_FREQ,
        BAUD_RATE => BAUD_RATE
    )
    port map (
        clk       => clk,
        reset_n   => reset_n,
        rx        => uart_rx,
        rx_data   => rx_data,
        rx_valid  => rx_valid
    );

    -- UART TX instance (for acknowledgments/status)
    uart_tx_inst : UART_TX_COMP
    generic map (
        CLK_FREQ  => CLK_FREQ,
        BAUD_RATE => BAUD_RATE
    )
    port map (
        clk       => clk,
        reset_n   => reset_n,
        tx        => uart_tx,
        tx_data   => tx_data,
        tx_start  => tx_start,
        tx_new_frame  => tx_new_frame,
        tx_busy   => tx_busy
    );

    -- Main state machine
    process(clk)
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                state <= IDLE_STATE;
                byte_counter <= 0;
                pixel_counter <= 0;
                rgb_counter <= 0;
                output_pixel_counter <= 0;
                frame_ready_int <= '0';
                error_flag_int <= '0';
                header_error_count <= 0;
                output_active <= '0';
                pixel_buffer <= (others => '0');
				echo <= '0';
            else
                -- Default values
                pixel_ready <= '0';
				echo <= '0';
                
                case state is
                    when IDLE_STATE =>
                        byte_counter <= 0;
                        pixel_counter <= 0;
                        rgb_counter <= 0;
                        output_pixel_counter <= 0;
                        frame_ready_int <= '0';
                        error_flag_int <= '0';
                        output_active <= '0';
                        
                        if rx_valid = '1' then
                            if rx_data = HEADER_BYTE_1 then
                                state <= WAIT_HEADER_2;
                            else
								if rx_data = ECHO_BYTE then
									echo <= '1';
								end if;
                                header_error_count <= header_error_count + 1;
                                   state <= IDLE_STATE;
                            end if;
                        end if;
                    
                    when WAIT_HEADER_2 =>
                        if rx_valid = '1' then
                            if rx_data = HEADER_BYTE_2 then
                                state <= WAIT_HEADER_3;
                            else
                                state <= IDLE_STATE;
                                header_error_count <= header_error_count + 1;
                            end if;
                        end if;
                    
                    when WAIT_HEADER_3 =>
                        if rx_valid = '1' then
                            if rx_data = HEADER_BYTE_3 then
                                state <= RECEIVE_PIXEL_DATA;
                                header_error_count <= 0; -- Reset error count on successful header
                            else
                                state <= IDLE_STATE;
                                header_error_count <= header_error_count + 1;
                            end if;
                        end if;
                    
                    when RECEIVE_PIXEL_DATA =>
                        if rx_valid = '1' then
						
							pixel_buffer <= rx_data;
							pixel_ready <= '1';
                            
                            -- Update counters
                            byte_counter <= byte_counter + 1;
							
							last_column <= current_column;
							last_row <= current_row;
							if current_column = FRAME_WIDTH-1 then
								current_column <= 0;
								if current_row < FRAME_HEIGHT-1 then
									current_row <= current_row + 1;
								end if;
							else
								current_column <= current_column + 1;
							end if;
                            
                            -- Check if frame complete
                            if byte_counter >= TOTAL_BYTES - 1 then
                                state <= FRAME_COMPLETE;
                                output_pixel_counter <= 0;
                                output_active <= '1';
                                output_delay_counter <= 0;
                                current_column <= 0;
                                current_row <= 0;
                            end if;
                        end if;
                    
                    when FRAME_COMPLETE =>
                        frame_ready_int <= '1';
                        state <= IDLE_STATE;
                    
                    when others =>
                        state <= IDLE_STATE;
                end case;
            end if;

            -- output stream data
            rxStream.Data_Valid <= rxStream_Reg.Data_Valid;
            rxStream.Column <= rxStream_Reg.Column;
            rxStream.Row <= rxStream_Reg.Row;
            rxStream.Filter <= rxStream_Reg.Filter;
            rxData <= rxData_Reg;
        end if;
    end process;

    -- CNN streaming output
    rxStream_Reg.Data_CLK <= clk;
    rxStream.Data_CLK <= clk;
    rxStream_Reg.Data_Valid <= pixel_ready;
    rxStream_Reg.Column <= last_column;
    rxStream_Reg.Row <= last_row;
    rxStream_Reg.Filter <= 0;

    -- Convert 8-bit RGB to 7-bit for CNN (shift right by 1)
    rxData_Reg(0) <= to_integer(unsigned(pixel_buffer(7 downto 0))); -- R (7-bit)
    rxData_Reg(1) <= to_integer(unsigned(pixel_buffer(7 downto 0)));  -- G (7-bit)
    rxData_Reg(2) <= to_integer(unsigned(pixel_buffer(7 downto 0)));   -- B (7-bit)
	
	process(clk)
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                tx_start <= '0';
				tx_new_frame <= '0';
				tx_data <= x"00";
            else
                -- Default values
                tx_start <= '0';
				tx_new_frame <= '0';
				
				if txStream.Data_Valid = '1' then
					if txStream.Row = 0 and txStream.Column = 0 then
						tx_new_frame <= '1';
					end if;
					tx_data <= std_logic_vector(to_unsigned(txData(0),8));
					tx_start <= '1';
				else
					if echo = '1' then
						tx_data <= x"42";
						tx_start <= '1';
					end if;
					if beat = '1' then
						tx_data <= debug_state;
						--tx_data <= x"43";
						--tx_start <= '1';
					end if;
				end if;
				
				
            end if;
        end if;
    end process;
	
	process(clk)
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                beat <= '0';
				last_heart_beat <= heartbeat;
				debug_buf <= x"0";
            else
                -- Default values
                beat <= '0';
				debug_buf <= debug_buf;
				
				--if beat_count < 10 then
					if heartbeat /= last_heart_beat then
						
						if heartbeat = '0' then
							beat <= '1';
							debug_buf <= x"4";
						end if;
					end if;
				--end if;
				
				last_heart_beat <= heartbeat;
				
            end if;
        end if;
    end process;

    -- Output assignments
    frame_ready <= frame_ready_int;
    error_flag <= error_flag_int;
    bytes_received <= std_logic_vector(to_unsigned(byte_counter, 16));
	
	current_state <= debug_buf;
	
    -- Debug state output
    with state select debug_state <=
        x"30" when IDLE_STATE,
        x"31" when WAIT_HEADER_1,
        x"32" when WAIT_HEADER_2,
        x"33" when WAIT_HEADER_3,
        x"34" when RECEIVE_PIXEL_DATA,
        x"35" when OUTPUT_PIXEL,
        x"36" when FRAME_COMPLETE,
        x"FF" when ERROR_STATE,
        x"3F" when others;
		
	-- uart_tx <= uart_rx;

end rtl;