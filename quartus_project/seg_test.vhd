--------------------------------------------------------------------------------
-- ONEAI CNN with UART Interface for AXC3000 FPGA
--------------------------------------------------------------------------------
-- This top-level design integrates:
-- - UART Interface for receiving 128x128 RGB image data
-- - ONEAI CNN for image processing/classification
-- - Status LEDs and control interface
-- - Clock management and reset handling
--
-- UART Protocol:
-- - Baud Rate: 115200
-- - Frame Format: Header (0xFF 0xAA 0x55) + 49,152 RGB bytes
-- - Response: Byte with class identified by CNN.
--
-- Pin Assignments:
-- - UART RX/TX: Connect to external UART interface
-- - LEDs: Status indication
-- - Push Button: Manual reset/control
--------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.CNN_Config_Package.all;

entity seg_test is
    port (
        -- Reset & Clock
        clk_in      		: in  std_logic;        -- Clock input. Adapt UART entity to Frequency!
        
        -- GPIO
        io96_3a_pb0         : in  std_logic;        -- Push button 0 - reset
        io96_3a_led0        : out std_logic;        -- Status LED 0 - heartbeat
        
        -- UART Interface
        uart_rx_pin         : in  std_logic;        -- UART receive
        uart_tx_pin         : out std_logic;         -- UART transmit
        -- spy UART Interface
        spy_rx_pin         : in  std_logic;        -- UART receive
        spy_tx_pin         : out std_logic         -- UART transmit
    );
end seg_test;

architecture rtl of seg_test is

    -- =================================================================
    -- Component Declarations
    -- =================================================================
	
	component pll_25_to_100 is
		port (
			refclk   : in  std_logic; 		-- clk
			locked   : out std_logic;        -- export
			rst      : in  std_logic; 		-- reset
			outclk_0 : out std_logic         -- clk
		);
	end component pll_25_to_100;
    
    component UART_Image_Interface is
		generic (
            CLK_FREQ  	: natural := 100_000_000;
			BAUD_RATE   : natural := 3_000_000
        );
        port (
            -- Clock and Reset
            clk             : in  std_logic;
            reset_n         : in  std_logic;
            
            -- UART Physical Interface
            uart_rx         : in  std_logic;
            uart_tx         : out std_logic;
            
            -- CNN Streaming Interface Output
            rxStream        : out CNN_Stream_T;
            rxData          : out CNN_Values_T(2 downto 0);
            
            -- CNN Streaming Interface Input
            txStream        : in  CNN_Stream_T;
            txData          : in  CNN_Values_T(0 downto 0)
        );
    end component;
    
    component CNN is
        port (
            -- Input Stream from UART
            iStream         : in  CNN_Stream_T;
            iData_1         : in  CNN_Values_T(2 downto 0);
            
            -- Output Stream
            oStream_1       : out CNN_Stream_T;
            oData_1         : out CNN_Values_T(0 downto 0)--;
            --oCycle_1        : out NATURAL
        );
    end component;
    
    -- =================================================================
    -- Internal Signals
    -- =================================================================
    
    -- Clock and Reset Management
    signal system_clk           : std_logic;                        -- Main system clock (100MHz)
    signal pll_clk           	: std_logic;                        -- PLL out clock (100MHz)
    signal system_reset_n       : std_logic;                        -- Synchronized reset
    signal reset_sync           : std_logic;    -- Reset synchronizer
    
    -- UART Interface Signals
    signal uart_rx              : std_logic;
    signal uart_tx              : std_logic;
    
    -- Status and Control
    signal led_counter          : unsigned(25 downto 0) := (others => '0');  -- LED blink counter
    signal heartbeat_led        : std_logic;                                  -- Heartbeat indicator
    signal manual_reset         : std_logic;                                  -- Manual reset from button
    signal pll_reset         	: std_logic;                                  -- inverse of manual reset button
    
    -- Button debounce
    signal button_debounce      : unsigned(19 downto 0) := (others => '0');
    signal button_sync          : std_logic := '1';
    
    -- Streaming interface bundles
    signal rxStream      		: CNN_Stream_T;
    signal rxData_bundle        : CNN_Values_T(2 downto 0);
    signal txStream      		: CNN_Stream_T;
    signal cnn_output_data      : CNN_Values_T(0 downto 0);
    signal cnn_output_cycle     : NATURAL;

begin

    -- =================================================================
    -- Clock and Reset Management
    -- =================================================================
	
	pll0 : component pll_25_to_100
		port map (
			refclk   => clk_in,   		--  refclk.clk
			locked   => reset_sync,   	--  locked.export
			rst      => pll_reset,   	--  reset.reset
			outclk_0 => pll_clk  		--  outclk0.clk
		);
	
	system_clk <= pll_clk;
    
    -- Manual reset from push button (active low, debounced)
    process(system_clk)
    begin
        if rising_edge(system_clk) then
            if io96_3a_pb0 = '0' then
                if button_debounce < x"FFFFF" then
                    button_debounce <= button_debounce + 1;
                else
                    button_sync <= '0';
                end if;
            else
                button_debounce <= (others => '0');
                button_sync <= '1';
            end if;
        end if;
    end process;
    
    manual_reset <= button_sync;
    system_reset_n <= manual_reset;
	pll_reset <= not manual_reset;
    
    
    -- =================================================================
    -- UART Image Interface Instance
    -- =================================================================
    
    uart_inst : UART_Image_Interface
        generic map (
            CLK_FREQ  	=> 100_000_000,
			BAUD_RATE   => 3_000_000
        )
        port map (
            -- Clock and Reset
            clk             => system_clk,
            reset_n         => system_reset_n,
            
            -- UART Physical Interface
            uart_rx         => uart_rx,
            uart_tx         => uart_tx,
            
            -- CNN Streaming Interface Output
            rxStream        => rxStream,
            rxData          => rxData_bundle,
            
            -- CNN Streaming Interface Input
            txStream        => txStream,
            txData          => cnn_output_data
        );
    
    -- =================================================================
    -- ONEAI CNN Instance
    -- =================================================================
    
    oneai_cnn_inst : CNN
        port map (
            -- Input Stream from UART
            iStream         => rxStream,
            iData_1         => rxData_bundle,
            
            -- Output Stream
            oStream_1       => txStream,
            oData_1         => cnn_output_data--,
            --oCycle_1        => cnn_output_cycle
        );
    
    -- =================================================================
    -- Status and LED Control
    -- =================================================================
    
    -- Heartbeat counter for LED blinking
    process(system_clk, system_reset_n)
    begin
        if system_reset_n = '0' then
            led_counter <= (others => '0');
        elsif rising_edge(system_clk) then
            led_counter <= led_counter + 1;
        end if;
    end process;
    
    -- Heartbeat LED (blinks at ~1Hz when system is running)
    heartbeat_led <= led_counter(24);  -- ~0.75Hz blink rate at 100MHz
    
    -- =================================================================
    -- GPIO Assignments
    -- =================================================================
    
    -- LED assignments
    io96_3a_led0 <= heartbeat_led;     -- Heartbeat - system alive
    
    -- UART pin assignments
    uart_tx_pin <= uart_tx;
    uart_rx <= uart_rx_pin;
	
    -- UART pin assignments
    spy_tx_pin <= uart_tx;
    --uart_rx <= uart_rx_pin;

end rtl;
