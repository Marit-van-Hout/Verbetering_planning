

def plot_schedule(scheduled_orders):
    """Plots a Gantt chart of the scheduled orders

    Args:
        scheduled_orders (dict): every order, their starting time, end time, on which machine and set-up time
    """    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = 0
    
    # Colors for visualization
    color_map = {
         '400': 'blue',
         '401': 'yellow',
    }
    
    for machine, orders in scheduled_orders.items():
        y_pos += 1  # Voor elke machine
        for order in orders:
            order_color = order['colour']
            processing_time = order['end_time'] - order['start_time'] - order['setup_time']
            setup_time = order['setup_time']
            start_time = order['start_time']
            
            # Controleer of de kleur aanwezig is in de color_map
            if order_color in color_map:
                color = color_map[order_color]
            else:
                color = 'black'  # Default color als de kleur niet bestaat in color_map
            
            # Teken verwerkingstijd
            ax.barh(y_pos, processing_time, left=start_time + setup_time, color=color, edgecolor='black')
            ax.text(start_time + setup_time + processing_time / 2, y_pos, f"Order {order['order']}", 
                    ha='center', va='center', color='black', rotation=90)

            # Teken setup tijd
            if setup_time > 0:
                ax.barh(y_pos, setup_time, left=start_time, color='gray', edgecolor='black', hatch='//')
    
    ax.set_yticks(range(1, len(scheduled_orders) + 1))
    ax.set_yticklabels([f"Machine {m}" for m in scheduled_orders.keys()])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title('Gantt Chart for Paint Shop Scheduling')
    plt.show()
