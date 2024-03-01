


ALTER TABLE `electric_cars`.`total_carsales` 
CHANGE COLUMN `Row_Id` `Row_Id` INT NOT NULL ,
ADD PRIMARY KEY (`Row_Id`);
;

SELECT Row_Id, Country, Year, Electric_cars_sold, Nonelectric_car_sales
FROM total_carsales
ORDER BY 1,2

-- % Of ELectric Cars sold total
ALTER TABLE electric_cars.total_carsales
ADD COLUMN Electric_Cars_Sold_Percentage DECIMAL(5, 2) NOT NULL;

SELECT Row_Id, Country, Year, Electric_cars_sold, Nonelectric_car_sales, (Electric_cars_sold/Nonelectric_car_sales)*100 as Electric_Cars_Sold_Percentage
FROM total_carsales
ORDER BY 1,2



-- % Of Electric Cars Sold that are FULL Electric & Not hybrid

ALTER TABLE `electric_cars`.`full_battery_electric_sold` 
CHANGE COLUMN `Row_Id` `Row_Id` INT NOT NULL ,
ADD FOREIGN KEY (`Row_Id`)
REFERENCES total_carsales(Row_Id);


SELECT Row_ID, Country, Year, FULL_Battery_Electric_sold
FROM full_battery_electric_sold
ORDER BY 1,2

ALTER TABLE electric_cars.full_battery_electric_sold
ADD COLUMN Full_Battery_Cars_Sold_Percentage INT NOT NULL;



SELECT 
total_carsales.Row_ID, 
total_carsales.Country, 
total_carsales.Year, 
total_carsales.Electric_cars_sold, 
total_carsales.Nonelectric_car_sales, 
(Electric_cars_sold/Nonelectric_car_sales)*100 as Electric_Cars_Sold_Percentage, FULL_Battery_Electric_sold, 
(FULL_Battery_Electric_sold/Electric_cars_sold)* 100 as Full_Battery_Cars_Sold_Percentage
FROM total_carsales
JOIN full_battery_electric_sold
ON total_carsales.Row_ID = full_battery_electric_sold.Row_ID
ORDER BY 1,2


-- Looking at top 5 countries countries with largest number of electric cars sold, here I used SUM instead of MAX because MAX is an aggregate function and could give maximum value for a specific combination of country and year
SELECT
country,
SUM(electric_cars_sold) As Total_Electric_Cars_Sold
FROM  total_carsales
WHERE
  country NOT IN ('World', 'European Union (27)', 'Europe')
GROUP BY country
ORDER BY Total_Electric_Cars_Sold DESC
LIMIT 5;

-- Calculating the Percentage of Electric Cars Sold for Each Country and Year Compared to the Total for That Year

SELECT
  Country,
  Year,
  Electric_cars_sold,
  Nonelectric_car_sales,
  100 * (Electric_cars_sold / SUM(Electric_cars_sold) OVER (PARTITION BY Year)) AS Electric_Cars_Percentage
FROM
  total_carsales
WHERE
  Country NOT IN ('World', 'European Union (27)', 'Europe')
ORDER BY
  Country, Year;
  
  
  
  
  -- Calculating Yearly Growth Rate of Electric Cars Sold:
  
SELECT
  Year,
  Country,
  Electric_cars_sold,
  LAG(Electric_cars_sold) OVER (PARTITION BY Country ORDER BY Year) AS Previous_Year_Electric_Cars,
  100 * ((Electric_cars_sold - LAG(Electric_cars_sold) OVER (PARTITION BY Country ORDER BY Year)) / LAG(Electric_cars_sold) OVER (PARTITION BY Country ORDER BY Year)) AS Growth_Rate_percent
FROM
  total_carsales
WHERE
  Country NOT IN ('World', 'European Union (27)', 'Europe')
ORDER BY
  Country, Year;





