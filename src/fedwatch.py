import pandas as pd
import numpy as np
import math
from datetime import datetime
from calendar import monthrange
import pandas_datareader as pdr
import inspect

import pandas as pd
import math

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from datetime import datetime
from calendar import monthrange
from dateutil.relativedelta import relativedelta
import holidays


class FOMC():

    def __init__(self, watch_date, fomc_dates, num_upcoming):

        # Check and initialize the calculation date
        if isinstance(watch_date, str):
            self.watch_date = datetime.strptime(watch_date, '%Y-%m-%d')
        elif isinstance(watch_date, datetime):
            self.watch_date = watch_date
        else:
            raise ValueError(
                "Invalid format for watch_date. It should be a string in 'yyyy-mm-dd' format or a datetime object.")

        # Check and initialize FOMC dates list
        if all(isinstance(date, datetime) for date in fomc_dates):
            self.fomc_dates = fomc_dates
        elif all(isinstance(date, str) for date in fomc_dates):
            self.fomc_dates = [datetime.strptime(date, '%Y-%m-%d') for date in fomc_dates]
        else:
            raise ValueError(
                "Invalid format for fomc_dates. It should be a list of datetime objects or a list of strings in 'yyyy-mm-dd' format.")

        # Sort fomc_dates list in ascending order
        self.fomc_dates = sorted(self.fomc_dates)

        # Initialize the requested number of upcoming FOMC meetings
        self.num_upcoming = num_upcoming

        # Initialize lists
        self.month_list = self.generate_month_list()
        self.contract_list = self.generate_contract_list()
        self.meeting_list = self.generate_meeting_list()
        self.order_list = self.generate_order_list()

        # sort fomc_dates
        self.summary = pd.DataFrame({'Contract': self.contract_list,
                                     'Meeting': self.meeting_list,
                                     'Order': self.order_list},
                                    index=self.month_list).rename_axis('YYYY-MM')

    def starting_no_fomc_month(self) -> tuple[int, int]:
        """
        Finds the first past month without an FOMC meeting.

        :return: (tuple[int, int]) A tuple containing the year and month of the target month.
        """

        # Format FOMC date to YYYY-MM to avoid confusion with day numbers
        fomc_dates_formatted = [date.strftime('%Y-%m') for date in self.fomc_dates if
                                date.strftime('%Y-%m') <= self.watch_date.strftime('%Y-%m')]

        target_month = self.watch_date
        starting_no_fomc = None

        while target_month.strftime('%Y-%m') >= fomc_dates_formatted[0]:
            if target_month.strftime('%Y-%m') not in fomc_dates_formatted:
                starting_no_fomc = target_month
                break;
            else:
                target_month = target_month - relativedelta(months=1)

        if starting_no_fomc == None:
            message = 'Starting No-FOMC Month not found! There might be an issue with the provided list of scheduled FOMC meetings.'
            raise ValueError(message)

        return starting_no_fomc.year, starting_no_fomc.month

    def ending_no_fomc_month(self) -> tuple[int, int]:
        """
        Finds the first upcoming month witout FOMC meeting, after the requested number of FOMC meetings.
        If calculation date to be in a month with FOMC meeting, does not count that month if that month's meeting is held.

        :return: (Tuple[int, int]) A tuple containing the year and month of the target month.
        """

        fomc_dates_formatted = [date.strftime('%Y-%m') for date in self.fomc_dates if date >= self.watch_date]

        target_month = self.watch_date
        ending_no_fomc = None

        fomc_counter = 0
        while target_month.strftime('%Y-%m') <= fomc_dates_formatted[-1]:
            if target_month.strftime('%Y-%m') in fomc_dates_formatted:
                fomc_counter += 1
            else:
                ending_no_fomc = target_month
                if fomc_counter >= self.num_upcoming:
                    break;

            target_month = target_month + relativedelta(months=1)

        if fomc_counter < self.num_upcoming:
            message = f'Number of FOMC meetings taken into account is {fomc_counter}, for {self.num_upcoming} meetings, extend the list of scheduled FOMC meetings!'
            raise ValueError(message)

        if ending_no_fomc == None:
            message = 'Ending No-FOMC Month not found! There might be an issue with the provided list of scheduled FOMC meetings.'
            raise ValueError(message)

        return ending_no_fomc.year, ending_no_fomc.month

    def generate_month_list(self):
        """
        Generates a list of months in the YYYY-MM format, which corresponds to the months
        for which futures contracts are required for FedWatch algorithm calculations.

        Before using this method, make sure to call "starting_no_fomc_month()" and "ending_no_fomc_month()"
        to find the start and end month.

        :return: (list of str) A list of months in YYYY-MM format.
        """

        start_month = self.starting_no_fomc_month()
        end_month = self.ending_no_fomc_month()

        month_list = pd.date_range(start=f"{start_month[0]}-{start_month[1]:02d}",
                                   end=f"{end_month[0]}-{end_month[1]:02d}",
                                   freq='MS')

        month_list = [date.strftime('%Y-%m') for date in month_list]

        return month_list

    def generate_contract_list(self):
        """
        Generates a list of Fed Funds futures contract symbols according to the CME naming convention,
        which are required for FedWatch algorithm calculations

        :return: (list of str) CME fed funds futures contratcs symbols.
        """

        # Generate CME fed funds futures contracts month code
        cme_month_codes = {
            1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
            7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
        }

        # Separate year and month from the month list
        year_month = [date.split('-') for date in self.month_list]

        # Create a list of contracts
        contract_list = ['ZQ' + cme_month_codes[int(month)] + year[-2:] for year, month in year_month]

        return contract_list

    def generate_meeting_list(self):
        """
        Generates a list of FOMC meeting dates according to the month list,
        which includes meeting date in YYYY-MM-DD string format for month
        with FOMC meeting and "No FOMC" for months without meeting.

        :return: (list of str) FOMC meeting date in YYYY-MM-DD format or "No FOMC".
        """

        # Create a new list based on original_dates
        fomc_meetings = []

        for date in self.month_list:
            year_month = date.split('-')
            matching_dates = [f"{date.strftime('%Y-%m-%d')}" for date in self.fomc_dates if
                              date.year == int(year_month[0]) and date.month == int(year_month[1])]

            if matching_dates:
                fomc_meetings.append(matching_dates[0])
            else:
                fomc_meetings.append('No FOMC')

        return fomc_meetings

    def generate_order_list(self):
        """
        Generates a list of FOMC meeting orders based on the month list and watch_date.
        For months with FOMC meetings, the integers represent the number of upcoming or past meetings relative to watch_date.

        :return: (list of int) 0 for months without FOMC meetings and integers for months with FOMC meetings.
        """
        # Extract year and month from the calculation date
        calc_yr, calc_mn = self.watch_date.year, self.watch_date.month

        # Find the month index of calculation date in month_list
        idx = next((i for i, month in enumerate(self.month_list) if month == f"{calc_yr}-{calc_mn:02d}"), None)

        # Create upcoming and past meetings list
        if self.meeting_list[idx] == 'No FOMC' or datetime.strptime(self.meeting_list[idx],
                                                                    '%Y-%m-%d') <= self.watch_date:
            fomc_list_bwd = self.meeting_list[:idx + 1]
            fomc_list_bwd.reverse()
            fomc_list_fwd = self.meeting_list[idx + 1:]
        else:
            fomc_list_bwd = self.meeting_list[:idx]
            fomc_list_bwd.reverse()
            fomc_list_fwd = self.meeting_list[idx:]

        # Create fomc upcoming meetings order list
        fomc_order_fwd = []
        meeting_counter = 1
        for date in fomc_list_fwd:
            if date == 'No FOMC':
                fomc_order_fwd.append(0)
            else:
                fomc_order_fwd.append(meeting_counter)
                meeting_counter += 1

        # Create fomc past meetings order list
        fomc_order_bwd = []
        meeting_counter = -1
        for date in fomc_list_bwd:
            if date == 'No FOMC':
                fomc_order_bwd.append(0)
            else:
                fomc_order_bwd.append(meeting_counter)
                meeting_counter -= 1
        fomc_order_bwd.reverse()

        return fomc_order_bwd + fomc_order_fwd

    def plot_fomc_calendar(self):
        """
        Plots a calendar that includes months for which their relevant fed funds futures contract data are required for FedWatch calculations.
        Highlights the calculation day and FOMC meetings on the calendar.

        :return: (matplotlib figure) Matplotlib figure of the calendar.
        """

        # internal ploting function - label_month
        def label_month(year, month, ax, i, j, cl="black"):
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            mn_code = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
            month_label = f"{months[month - 1]} {year} - ZQ{mn_code[month - 1]}{str(year)[-2:]}"
            ax.text(i, j, month_label, color=cl, va="center", fontsize=11)

        # internal ploting function - label_weekday
        def label_weekday(ax, i, j, cl="black"):
            x_offset_rate = 1
            for weekday in ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]:
                ax.text(i, j, weekday, ha="center", va="center", color=cl)
                i += x_offset_rate

        # internal ploting function - label_day
        def label_day(ax, day, i, j, cl="black"):
            ax.text(i, j, int(day), ha="center", va="center", color=cl)

        # internal ploting function - check_color_day
        def check_color_day(year, month, day, weekday):
            if (year, month, day) in holiday_list:
                return "red"  # holidays
            if weekday == 6:  # Sunday
                return "red"
            if weekday == 5:  # Saturday
                return "red"
            return "black"

        # internal ploting function - check_fill_day
        def check_fill_day(year, month, day, weekday):
            if (year, month, day) in fillday_list:
                return True

        # internal ploting function - check_calc_day
        def check_calc_day(year, month, day, weekday):
            if (year, month, day) == (self.watch_date.year, self.watch_date.month, self.watch_date.day):
                return True

        # internal ploting function - fill_box
        def fill_box(ax, i, j, edgecolor, facecolor):
            ax.add_patch(
                patches.Rectangle(
                    (i - 0.5, j - 0.5),
                    1,
                    1,
                    edgecolor=edgecolor,
                    facecolor=facecolor,
                    alpha=0.3,
                    fill=True,
                )
            )

        # internal ploting function - month_calendar
        def month_calendar(ax, year, month, fill):
            date = datetime(year, month, 1)
            weekday, num_days = monthrange(year, month)
            x_start = 1 - 0.5
            y_start = 5 + 0.5
            x_offset_rate = 1
            y_offset = -1
            label_month(year, month, ax, x_start, y_start + 2)
            label_weekday(ax, x_start, y_start + 1)
            j = y_start
            for day in range(1, num_days + 1):
                i = x_start + weekday * x_offset_rate
                color = check_color_day(year, month, day, weekday)
                if fill and check_fill_day(year, month, day, weekday):
                    fill_box(ax, i, j, "blue", "darkblue")
                if fill and check_calc_day(year, month, day, weekday):
                    fill_box(ax, i, j, "red", "darkred")
                label_day(ax, day, i, j, color)
                weekday = (weekday + 1) % 7
                if weekday == 0:
                    j += y_offset

        # FOMC meeting days as fill day list
        fillday_list = []
        for date_str in self.meeting_list:
            if date_str != 'No FOMC':
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                date_tuple = (date_obj.year, date_obj.month, date_obj.day)
                fillday_list.append(date_tuple)

        # Create holiday list
        holiday_list = []

        years = [int(date_str.split("-")[0]) for date_str in self.month_list]
        us_holidays = holidays.US(years=years)

        for date, name in sorted(list(us_holidays.items())):
            holiday_list.append((date.year, date.month, date.day))

        # Initializing number of rows, columns and size of the calendar
        ncol = 4
        nrow = math.ceil(len(self.month_list) / 4)
        figsize = (15, nrow * 3)
        fig, axs = plt.subplots(figsize=figsize, nrows=nrow, ncols=ncol)

        # Set title of the plot
        watch_date_str = self.watch_date.strftime('%B %d, %Y')
        fig.suptitle(f"FOMC Calendar: on {watch_date_str} for {self.num_upcoming} Upcoming Meetings",
                     fontsize=14, fontweight='bold', y=1.03)

        # Add a footnote to the plot
        plt.figtext(
            0.03, -0.01,
            r'$\bf{Note}$: The calendar includes the months for which corresponding Fed Funds Futures contracts pricing data '
            f'is required for FedWatch calculations. It starts with a no-FOMC month and continues for the requested \nnumber '
            f'of upcoming meetings, looking forward from the watch date, until another no-FOMC month. FOMC meeting days are '
            f'indicated with blue shaded boxes, and the red shaded box represents the watch day.',
            fontsize=10, horizontalalignment='left')

        fig.tight_layout();

        # Start populating the plot!
        ax_counter = 0
        for ax in axs.reshape(-1):
            # Create 7*7 grid for each subplot
            ax.axis([0, 7, 0, 7])
            ax.axis("on")
            ax.grid(True)

            # Hide x-axis ticks and lables for each subplot
            for tick in ax.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

            # Hide y-axis ticks and lables for each subplot
            for tick in ax.yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

            # Hide subplot if it is not in the list of months, otherwise populate it
            if ax_counter >= len(self.month_list):
                ax.set_visible(False)
            else:
                date_obj = datetime.strptime(self.month_list[ax_counter], "%Y-%m")
                month_calendar(ax, year=date_obj.year, month=date_obj.month, fill=True)
            ax_counter += 1

        return fig


class FedWatch():

    def __init__(self, watch_date, num_upcoming, fomc_dates, user_func, **kwargs):

        self.fomc_data = FOMC(watch_date, fomc_dates, num_upcoming)

        # Check if user_func accepts 'symbol' as its first argument
        argspec = inspect.getfullargspec(user_func)
        if argspec.args and argspec.args[0] != 'symbol':
            raise ValueError(
                f"The user_func must have 'symbol' as its first argument, '{argspec.args[0]}' is not acceptable as the first argument.")
        else:
            self.user_func = user_func
            # Check if 'symbol' is in kwargs
            if 'symbol' in kwargs:
                raise ValueError(
                    f"The 'symbol' argument should not be passed via **kwargs. Pass '{user_func.__name__}' arguments except 'symbol'.")
            else:
                self.user_args = kwargs

        # Initizlize rate expectation
        self.rate_expectations = None
        self.watch_rate_range = None

    def get_fff_history(self, symbol):
        """
        Uses user provided function to read OHLC data of fed fund futures contratcs. Then makes sure that the
        user function return a dataframe that meets the required criteria before proceeding with further analysis.
        """
        # Read symbol OHLC data using user provided function and arguments
        ohlc_df = self.user_func(symbol, **self.user_args)

        # Check if df is a DataFrame
        if not isinstance(ohlc_df, pd.DataFrame):
            raise ValueError(f"'{self.user_func.__name__}' does not return Pandas DataFrame for {symbol} contract.")

        # Check if 'Close' column exists in the DataFrame
        if 'Close' not in ohlc_df.columns:
            raise ValueError(
                f"'{self.user_func.__name__}' does not return a DataFrame with a 'Close' column for {symbol} contract.")

        # Check if 'Date' is either a column or an index in the DataFrame
        if 'Date' not in ohlc_df.columns and ohlc_df.index.name != 'Date':
            raise ValueError(
                f"'{self.user_func.__name__}' does not return a DataFrame with 'Date' as a column or an index for {symbol} contract.")

        # If 'Date' is not an index, set it as an index
        if ohlc_df.index.name == 'Date':
            # Check if the 'Date' column can be converted to datetime
            if pd.to_datetime(ohlc_df.index, errors='coerce').notna().all():
                ohlc_df.index = pd.to_datetime(ohlc_df.index, format='%Y-%m-%d')
            else:
                raise ValueError(
                    f"'{self.user_func.__name__}' does not return a DataFrame with a convertible 'Date' index for {symbol} contract.")

        # If 'Date' is not an index, set it as an index
        if ohlc_df.index.name != 'Date':
            # Check if the 'Date' column can be converted to datetime
            if 'Date' in ohlc_df.columns and pd.to_datetime(ohlc_df['Date'], errors='coerce').notna().all():
                ohlc_df['Date'] = pd.to_datetime(ohlc_df['Date'], format='%Y-%m-%d')
                ohlc_df.set_index('Date', inplace=True)
            else:
                raise ValueError(
                    f"'{self.user_func.__name__}' does not return a DataFrame with a convertible 'Date' column for {symbol} contract.")

        return ohlc_df

    def add_price_data(self):
        """
        Fills average price for each month in the month list with the closing price of the relevant contract
        on the watch date. If the watch date is holiday, then the nearest previous date with price data is used.

        For no FOMC months the start and end price are equal to average price, but for FOMC months start and end
        price are filled with zero to be replaced with appropriate value, later.

        :return: (3 float lists) p_start, p_avg and p_end, start and end price of FOMC months are zero!
        """
        # Define price lists
        p_start = []
        p_avg = []
        p_end = []

        # Extract watch date month and string date for comparisons
        watch_month = self.fomc_data.watch_date.strftime('%Y-%m')
        watch_date_str = self.fomc_data.watch_date.strftime('%Y-%m-%d')

        # Loop over the contract list to extract and fill price data
        for i in range(0, len(self.fomc_data.contract_list)):

            # Extract contract symbol, contract month and month type (FOMC or no-FOMC)
            contract_symbol = self.fomc_data.contract_list[i]
            contract_month = self.fomc_data.month_list[i]
            month_type = self.fomc_data.meeting_list[i]

            # Read price data of fed funds futures contract
            ohlc = self.get_fff_history(symbol=contract_symbol)

            # Fill price data
            if contract_month >= watch_month:  # If the contract not expired

                # Find and fill the average price, p_avg
                p_avg_i = ohlc[ohlc.index <= watch_date_str].iloc[-1, :]['Close']
                p_avg.append(p_avg_i)
            else:  # If the contract is expired

                # Find the month last day, as some data sources provide unreal price data after expiration
                yyyy_mm = datetime.strptime(contract_month, '%Y-%m')
                last_day = monthrange(yyyy_mm.year, yyyy_mm.month)[1]
                last_day_str = datetime(yyyy_mm.year, yyyy_mm.month, last_day).strftime('%Y-%m-%d')

                # Fill the average price using month last day
                p_avg_i = ohlc[ohlc.index <= last_day_str].iloc[-1, :]['Close']
                p_avg.append(p_avg_i)

                # For 'No FOMC' months, consider p_start and p_end equal to p_avg, otherwise consider 0.0
            if month_type == 'No FOMC':
                p_start.append(p_avg_i)
                p_end.append(p_avg_i)
            else:
                p_start.append(0.0)
                p_end.append(0.0)

        # Add price data to the fomc data summary
        self.fomc_data.summary['Pstart'] = p_start
        self.fomc_data.summary['Pavg'] = p_avg
        self.fomc_data.summary['Pend'] = p_end

        return p_start, p_avg, p_end

    def fill_price_data(self):
        """
        After adding start, average and end prices for No-FOMC months and average price for FOMC months, uses
        forward and backward propagation process to fill/calculate FOMC months start and end prices.

        :return: (3 float lists) p_start, p_avg and p_end
        """
        # Fill price data
        p_start, p_avg, p_end = self.add_price_data()

        # Forward propagation of price from No-FOMC month end price to the subseuent FOMC month start price
        for i in range(1, len(p_avg) - 1):  # ignore the first and last months as they are No-FOMC!

            # If FOMC month (p_start=0), fill with p_end of the previous month if it is No-FOMC
            if p_start[i] == 0.0 and p_end[i - 1] != 0.0:
                p_start[i] = p_end[i - 1]

            # If FOMC month (p_end=0), fill with p_start of the next month if it is No-FOMC
            if p_end[i] == 0.0 and p_start[i + 1] != 0.0:
                p_end[i] = p_start[i + 1]

        # Backward propagation of price to fill remaining zero prices in the list
        for i in range(len(p_avg) - 2, 0, -1):  # ignore the first and last months as they are No-FOMC!

            # If FOMC month and end price is still zero, fill with p_start of the next month (FOMC or No-FOMC)
            if p_end[i] == 0.0:
                p_end[i] = p_start[i + 1]

            # If FOMC month and start price is still zero, calculate using p_avg and p_end of the same month
            if p_start[i] == 0.0:
                # Extract meeting date, calculate m (days after FOMC meeting) and n (days before FOMC meeting)
                meeting_date = datetime.strptime(self.fomc_data.meeting_list[i], '%Y-%m-%d')
                days_no = monthrange(meeting_date.year, meeting_date.month)[1]
                m = days_no - meeting_date.day + 1
                n = days_no - m

                # Calculate start price using average and end price
                p_start[i] = (p_avg[i] - m / (m + n) * p_end[i]) / (n / (m + n))

        # Add price data to the fomc data summary
        self.fomc_data.summary['Pstart'] = p_start
        self.fomc_data.summary['Pavg'] = p_avg
        self.fomc_data.summary['Pend'] = p_end

        return p_start, p_avg, p_end

    def generate_binary_hike_info(self):
        """
        Having FOMC meetings summary data, generates binary hike sizes and relevant probabilities for FOMC meetings,
        which will be used later on, by other methods to generate cumulative hike sizes and relevant probabilities
        for upcoming FOMC meetings.
        """
        # add and fill price data
        self.fill_price_data()

        # Take a copy from FOMC meetings summary to build binary hike/cut values and probabilities
        bin_hike_df = self.fomc_data.summary.copy()

        # Filter out the requested number of upcoming FOMC meetings from the meeting list
        bin_hike_df = bin_hike_df[(bin_hike_df['Order'] > 0) & ((bin_hike_df['Order'] <= self.fomc_data.num_upcoming))]

        # Calculate the monthly change in the implied rate of contratcs
        bin_hike_df['Change'] = ((100 - bin_hike_df['Pend']) - (100 - bin_hike_df['Pstart'])) / 25 * 100

        # Using monthly change in implied rate, calculate binary hike sizes, H0 and H1
        bin_hike_df['H0'] = bin_hike_df['Change'].apply(lambda x: int(math.trunc(x) * 25))
        bin_hike_df['H1'] = bin_hike_df['Change'].apply(lambda x: int(math.trunc(x) * 25 + 25 * np.sign(x)))

        # Using monthly change in implied rate, calculate binary hike probabilities for H0 and H1, i.e. P0 and P1
        bin_hike_df['P0'] = bin_hike_df['Change'].apply(lambda x: 1 - (np.abs(x) - math.trunc(np.abs(x))))
        bin_hike_df['P1'] = bin_hike_df['Change'].apply(lambda x: (np.abs(x) - math.trunc(np.abs(x))))

        return bin_hike_df

    def generate_hike_info(self, rate_cols=True, watch_rate_range=None):
        """
        Generates hike/cut size with relevant probabilities for the requested number of upcoming FOMC meetings
        form the watch date.
        """

        def extract_binary_hike_info(group):
            """
            Takes binary hikes dataframe and extracts minimum and maximum hike sizes with relevant probabilities
            for each FOMC meeting in the form of numpy array. The output of this function is used to generate
            cumulative hike sizes and relevant probabilities, later on.
            """

            hike_info = {
                'hike_size': np.array(group[['H0', 'H1']].values[0]),
                'hike_prob': np.array(group[['P0', 'P1']].values[0])
            }

            return hike_info

        def calc_cum_info(lead_size, lag_size, lead_prob, lag_prob):
            """
            Takes two subsequent FOMC meetings binary hike information, [H0, H1] and [P0, P1] for lead and lag
            FOMC meetings and calculates all possible scenarios for the lag meeting with cumulative probability
            of happening.
            """
            size_list = lead_size[:, np.newaxis] + lag_size
            prob_list = lead_prob[:, np.newaxis] * lag_prob

            size_list_flat = size_list.flatten()
            prob_list_flat = prob_list.flatten()

            unique_size, indices = np.unique(size_list_flat, return_inverse=True)
            unique_prob = np.bincount(indices, weights=prob_list_flat)

            return unique_size, unique_prob

        # Find watch date rate range
        if rate_cols and not (watch_rate_range):

            # Extracts watch date and try to get the rate range from FRED database
            watch_date = self.fomc_data.watch_date

            try:
                # For watch dates after December 16, 2008, get lower and upper limit of target rate
                if watch_date >= datetime(2008, 12, 16):
                    ll = pdr.DataReader('DFEDTARL', 'fred', start=watch_date, end=watch_date).iloc[0, 0]
                    ul = pdr.DataReader('DFEDTARU', 'fred', start=watch_date, end=watch_date).iloc[0, 0]
                    watch_rate_range = (ll, ul)

                # For watch dates before December 16, 2008, get target rate and consider it as upper and lower limit
                else:
                    ll = ul = pdr.DataReader('DFEDTAR', 'fred', start=watch_date, end=watch_date).iloc[0, 0]
                    watch_rate_range = (ll, ul)

                self.watch_rate_range = watch_rate_range
            except:
                raise ValueError(
                    'Unable to get target rate limits from FRED database, please provide (ll, ul) for "watch_rate_range" or try again.')

        # Generate binary hike info and extract to a dictionary
        bin_hike_df = self.generate_binary_hike_info()
        binary_hike_info = bin_hike_df.groupby('Meeting').apply(extract_binary_hike_info)

        # Create CME FedWatch dataframe using the first upcoming FOMC meeting in the binary hike ifo dictionary
        meeting_date = binary_hike_info.keys()[0]
        meeting_size = binary_hike_info[meeting_date]['hike_size']
        meeting_prob = binary_hike_info[meeting_date]['hike_prob']

        meeting_data = {'FOMCDate': meeting_date}
        for size, prob in zip(meeting_size, meeting_prob):
            meeting_data[size] = [prob]

        fedwatch_df = pd.DataFrame(meeting_data).set_index('FOMCDate')

        # Initialize the loop
        lead_size = meeting_size
        lead_prob = meeting_prob

        # Add subsequent meetings data by calculating cumulative hike sizes and probabilities
        for i in range(1, len(binary_hike_info)):

            # Extract meeting data
            meeting_date = binary_hike_info.keys()[i]
            lag_size = binary_hike_info[meeting_date]['hike_size']
            lag_prob = binary_hike_info[meeting_date]['hike_prob']

            # Calculate cumulative info, size and probability
            meeting_size, meeting_prob = calc_cum_info(lead_size, lag_size, lead_prob, lag_prob)

            # Prepare cumulative info to be able to add it to fedwatch_df
            meeting_data = {'FOMCDate': meeting_date}
            for size, prob in zip(meeting_size, meeting_prob):
                meeting_data[size] = [prob]

            # Add meeting data to the dataframe
            fedwatch_df = pd.concat([fedwatch_df, pd.DataFrame(meeting_data).set_index('FOMCDate')]).fillna(0.0)

            # Update lead meeting info for the next round
            lead_size = meeting_size
            lead_prob = meeting_prob

        # Sort columns order in dataframe
        fedwatch_df.sort_index(axis=1, inplace=True)

        # Add watch date to the dataframe
        fedwatch_df['WatchDate'] = self.fomc_data.watch_date.strftime('%Y-%m-%d')
        fedwatch_df.reset_index(inplace=True)
        fedwatch_df.set_index(['WatchDate', 'FOMCDate'], inplace=True)

        # Modify column names if necessary
        if rate_cols:
            rate_columns = []
            rate_diff = watch_rate_range[1] - watch_rate_range[0]
            for column_name in fedwatch_df.columns:
                col_name_float = float(column_name / 100)
                if rate_diff == 0:
                    new_col_name = f"{watch_rate_range[0] + col_name_float:.2f}"
                else:
                    new_col_name = f"{watch_rate_range[0] + col_name_float:.2f}-{watch_rate_range[1] + col_name_float:.2f}"
                rate_columns.append(new_col_name)
            fedwatch_df.columns = rate_columns
            self.watch_rate_range = watch_rate_range

        # Update class data
        self.rate_expectations = fedwatch_df

        return fedwatch_df