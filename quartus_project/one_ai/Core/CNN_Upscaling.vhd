
--Description: -This component upscales the input image (e.g. 2x2 -> 4x4)
--Insertion:   -Connect the input data and stream signal with the input or previous layer

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.numeric_std.all; 
use work.CNN_Config_Package.all;


ENTITY CNN_Upscaling IS
    GENERIC (
        Input_Columns   : NATURAL := 28;      --Size in x direction of input
        Input_Rows      : NATURAL := 28;      --Size in y direction of input
        Input_Values    : NATURAL := 4;       --Number of Filters in previous layer
        Upscale_Columns : NATURAL := 2;       --Scale factor in x direction
        Upscale_Rows    : NATURAL := 2;       --Scale factor in y direction
        Expand_Delay    : NATURAL := 1;       --Cycles for output stream (1 = one value every cycle)
        Scaling_Type    : Upscaling_T := nearest; --Type of Upscaling (nearest, bilinear)
        Input_Cycles    : NATURAL := 1;       --Filter Cycles of previous convolution
        Filter_Delay    : NATURAL := 1;       --Cycles between Filters
        Row_Delay       : NATURAL := 0        --Cycles of pause between rows
    );
    PORT (
        CLK     : IN  STD_LOGIC;
        iStream : IN  CNN_Stream_T;
        iData   : IN  CNN_Values_T(Input_Values/Input_Cycles-1 downto 0);
        
        oStream : OUT CNN_Stream_T;
        oData   : OUT CNN_Values_T(Input_Values/Input_Cycles-1 downto 0) := (others => 0)
    );
END CNN_Upscaling;

ARCHITECTURE BEHAVIORAL OF CNN_Upscaling IS

    CONSTANT In_Values     : NATURAL := Input_Values/Input_Cycles;
    CONSTANT RAM_Bits      : NATURAL := (CNN_Value_Resolution+CNN_Value_Negative)*In_Values;
    CONSTANT RAM_Width     : NATURAL := Input_Columns*Input_Cycles*2;
    
    type RAM_T is array (RAM_Width-1 downto 0) of STD_LOGIC_VECTOR(RAM_Bits-1 downto 0);
    SIGNAL Buffer_RAM      : RAM_T := (others => (others => '0'));
    SIGNAL RAM_Addr_Out    : natural range 0 to RAM_Width - 1;
    SIGNAL RAM_Addr_In     : natural range 0 to RAM_Width - 1;
    SIGNAL RAM_Data_In     : std_logic_vector(RAM_Bits - 1 downto 0);
    SIGNAL RAM_Data_Out    : std_logic_vector(RAM_Bits - 1 downto 0);
    SIGNAL RAM_Enable      : STD_LOGIC := '1';
    
    SIGNAL oStream_Reg     : CNN_Stream_T;
  
BEGIN

  oStream.Data_CLK <= iStream.Data_CLK;
  RAM_Data_Out <= Buffer_RAM(RAM_Addr_Out);
  PROCESS (iStream)
    
  BEGIN
    IF (rising_edge(iStream.Data_CLK)) THEN
      IF (RAM_Enable = '1') THEN
        Buffer_RAM(RAM_Addr_In) <= RAM_Data_In;
      
      END IF;
    
    END IF;
  END PROCESS;
  PROCESS (iStream)
    VARIABLE Column_Reg       : NATURAL range 0 to Input_Columns-1 := 0;
    VARIABLE Row_Reg          : NATURAL range 0 to Input_Rows-1 := 0;
    VARIABLE Row_Cnt          : NATURAL range 0 to 1 := 0;
    VARIABLE Row_Rd_Cnt       : NATURAL range 0 to 1 := 0;
    VARIABLE Out_Row_Cnt      : NATURAL range 0 to Upscale_Rows      := Upscale_Rows;
    VARIABLE Out_Column_Cnt   : NATURAL range 0 to (Input_Columns*Upscale_Columns)-1 := (Input_Columns*Upscale_Columns)-1;
    VARIABLE Out_Column_Delay : NATURAL range 0 to Expand_Delay-1      := 0;
    VARIABLE Out_Filter_Cnt   : NATURAL range 0 to Input_Cycles-1    := Input_Cycles-1;
    VARIABLE Out_Filter_Delay : NATURAL range 0 to Filter_Delay-1    := Filter_Delay-1;
    VARIABLE Out_Row_Delay    : NATURAL := 0;
    VARIABLE Out_Row_Reg      : NATURAL range 0 to Input_Rows-1      := Input_Rows-1;
    VARIABLE Just_Reset       : BOOLEAN := FALSE;
  BEGIN
    IF (rising_edge(iStream.Data_CLK)) THEN
      Just_Reset := FALSE;
      IF (Column_Reg > iStream.Column OR (iStream.Data_Valid = '1' AND Out_Row_Cnt = Upscale_Rows)) THEN
        if Column_Reg > iStream.Column then
            Row_Rd_Cnt := Row_Cnt;
            Row_Cnt    := (Row_Cnt + 1) mod 2;
            Out_Row_Reg      := Row_Reg;
            Row_Reg    := iStream.Row;
            
            -- Only start outputting when we have a completed row or enough data
            Out_Row_Cnt      := 0;
            Out_Column_Cnt   := 0;
            Out_Column_Delay := 0;
            Out_Filter_Cnt   := 0;
            Out_Filter_Delay := 0;
            Out_Row_Delay    := 0;
            Just_Reset       := TRUE;
        end if;
      
      END IF;
      Column_Reg := iStream.Column;
      
      -- Only run counters if we are processing a valid Upscaling block
      IF (Out_Row_Cnt < Upscale_Rows AND NOT Just_Reset) THEN
          IF (Out_Filter_Delay < Filter_Delay-1) THEN
            Out_Filter_Delay := Out_Filter_Delay + 1;
          ELSIF (Out_Filter_Cnt < Input_Cycles-1) THEN
            Out_Filter_Delay := 0;
            Out_Filter_Cnt := Out_Filter_Cnt + 1;
          ELSIF (Out_Column_Delay < Expand_Delay-1) THEN
            Out_Column_Delay := Out_Column_Delay + 1;
          ELSIF (Out_Column_Cnt < (Input_Columns*Upscale_Columns)-1) THEN
            Out_Filter_Delay := 0;
            Out_Filter_Cnt := 0;
            Out_Column_Delay := 0;
            Out_Column_Cnt := Out_Column_Cnt + 1;
          ELSIF (Out_Row_Delay < Row_Delay) THEN
            -- Row delay before moving to next row
            Out_Row_Delay := Out_Row_Delay + 1;
          ELSE
            -- Row delay complete, move to next upscaled row
            Out_Filter_Delay := 0;
            Out_Filter_Cnt := 0;
            Out_Column_Delay := 0;
            Out_Column_Cnt := 0;
            Out_Row_Delay := 0;
            Out_Row_Cnt := Out_Row_Cnt + 1;
          
          END IF;
      END IF;
      RAM_Addr_Out <= (Row_Rd_Cnt*Input_Columns+(Out_Column_Cnt/Upscale_Columns))*Input_Cycles+Out_Filter_Cnt;
      
      if Out_Row_Cnt < Upscale_Rows then
        oStream_Reg.Row    <= Out_Row_Reg*Upscale_Rows+Out_Row_Cnt;
      else
        oStream_Reg.Row    <= Out_Row_Reg*Upscale_Rows+(Upscale_Rows-1);
      end if;

      oStream_Reg.Column <= Out_Column_Cnt;
      oStream_Reg.Filter <= Out_Filter_Cnt*In_Values;
      IF (Out_Filter_Delay = 0 AND Out_Column_Delay = 0 AND Out_Row_Delay = 0 AND Out_Row_Cnt < Upscale_Rows) THEN
        oStream_Reg.Data_Valid <= '1';
      ELSE
        oStream_Reg.Data_Valid <= '0';
      END IF;
      FOR i in 0 to In_Values-1 LOOP
        IF (CNN_Value_Negative = 0) THEN
          oData(i) <= TO_INTEGER(UNSIGNED(RAM_Data_Out((CNN_Value_Resolution*(i+1))-1 downto CNN_Value_Resolution*i)));
        ELSE
          oData(i) <=  TO_INTEGER(SIGNED(RAM_Data_Out(((CNN_Value_Resolution+1)*(i+1))-1 downto (CNN_Value_Resolution+1)*i)));
        END IF;
      END LOOP;
  
      oStream.Row        <= oStream_Reg.Row;
      oStream.Column     <= oStream_Reg.Column;
      oStream.Filter     <= oStream_Reg.Filter;
      oStream.Data_Valid <= oStream_Reg.Data_Valid;
      RAM_Addr_In  <= (Row_Cnt*Input_Columns+iStream.Column)*Input_Cycles+(iStream.Filter/In_Values);
      FOR i in 0 to In_Values-1 LOOP
        IF (CNN_Value_Negative = 0) THEN
          RAM_Data_In((CNN_Value_Resolution*(i+1))-1 downto CNN_Value_Resolution*i) <= STD_LOGIC_VECTOR(TO_UNSIGNED(iData(i), CNN_Value_Resolution));
        ELSE
          RAM_Data_In(((CNN_Value_Resolution+1)*(i+1))-1 downto (CNN_Value_Resolution+1)*i) <= STD_LOGIC_VECTOR(TO_SIGNED(iData(i), CNN_Value_Resolution+1));
        END IF;
      END LOOP;
    END IF;
  END PROCESS;
  
END BEHAVIORAL;